"""
Incremental geo-referenced mosaic ("the map").

How a photo is placed
---------------------
1. Its 4 corners are projected to the ground with the camera model and the
   drone's GPS/altitude/attitude (mapping/camera_model.py). For a flat ground
   and a pinhole camera the image -> ground mapping is exactly a homography,
   so roll/pitch/yaw are fully corrected (not just a rotated rectangle).
2. Refinement: GPS error is ~1-2 m, so neighbouring photos would not line up.
   We estimate the residual shift between the new photo and the map already
   built in the overlap with phase correlation (FFT, fast and robust on
   grass). Shifts are accepted only if confident and smaller than
   `max_refine_shift_m`, so a bad match can never move a photo far.
3. Exposure compensation: gain-match brightness in the overlap.
4. Seam selection: every map pixel keeps the photo pixel that was closest to
   that photo's centre (most vertical view, least distortion, least relief
   displacement). This is the classic nadir-Voronoi seamline used by
   orthomosaic software and costs one uint16 comparison per pixel.

The canvas is north-up, `resolution_m` metres per pixel, and grows as needed.
Memory: ~5 bytes/pixel; a 200 m x 200 m area at 5 cm = 16 Mpx = 80 MB.
"""
from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from mapping.camera_model import CameraModel
from mapping.gps_converter import LocalFrame

log = logging.getLogger("mosaic")
EMPTY = np.uint16(65535)


class GeoMosaic:
    def __init__(self, frame: LocalFrame, resolution_m: float = 0.05, max_canvas_px: int = 16000,
                 grow_margin_m: float = 40.0, refine: bool = True, max_refine_shift_m: float = 3.0,
                 min_refine_response: float = 0.08, exposure_comp: bool = True, gps_weight: float = 0.2):
        self.gps_weight = gps_weight
        self.frame = frame
        self.res = float(resolution_m)
        self.max_px = max_canvas_px
        self.margin = grow_margin_m
        self.refine = refine
        self.max_shift_px = max_refine_shift_m / self.res
        self.min_resp = min_refine_response
        self.exposure_comp = exposure_comp
        self.canvas: np.ndarray | None = None
        self.score: np.ndarray | None = None
        self.n_max = 0.0     # north coordinate of the top edge
        self.e_min = 0.0     # east coordinate of the left edge
        self.frames: dict[int, dict] = {}
        self._nadir_cache: dict = {}

    # --------------------------------------------------------------- canvas
    @property
    def shape(self):
        return (0, 0) if self.canvas is None else self.canvas.shape[:2]

    def ne_to_px(self, n, e):
        return (np.asarray(e) - self.e_min) / self.res, (self.n_max - np.asarray(n)) / self.res

    def px_to_ne(self, x, y):
        return self.n_max - np.asarray(y) * self.res, self.e_min + np.asarray(x) * self.res

    def _ensure(self, n_lo, n_hi, e_lo, e_hi) -> bool:
        m = self.margin
        if self.canvas is None:
            self.n_max, self.e_min = n_hi + m, e_lo - m
            h = int(math.ceil((n_hi - n_lo + 2 * m) / self.res))
            w = int(math.ceil((e_hi - e_lo + 2 * m) / self.res))
            if max(h, w) > self.max_px:
                log.error("first frame footprint too large for canvas")
                return False
            self.canvas = np.zeros((h, w, 3), np.uint8)
            self.score = np.full((h, w), EMPTY, np.uint16)
            return True
        h, w = self.shape
        top = max(0, int(math.ceil((n_hi - self.n_max) / self.res)))
        left = max(0, int(math.ceil((self.e_min - e_lo) / self.res)))
        bottom = max(0, int(math.ceil((self.n_max - h * self.res - n_lo) / self.res)))
        right = max(0, int(math.ceil((e_hi - (self.e_min + w * self.res)) / self.res)))
        if not (top or left or bottom or right):
            return True
        g = int(m / self.res)
        top, left, bottom, right = (v + g if v else 0 for v in (top, left, bottom, right))
        if max(h + top + bottom, w + left + right) > self.max_px:
            log.warning("canvas limit reached (%d px) - frame skipped", self.max_px)
            return False
        self.canvas = cv2.copyMakeBorder(self.canvas, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        self.score = cv2.copyMakeBorder(self.score, top, bottom, left, right, cv2.BORDER_CONSTANT, value=int(EMPTY))
        self.n_max += top * self.res
        self.e_min -= left * self.res
        return True

    def _nadir_map(self, w, h, cam: CameraModel) -> np.ndarray:
        key = (w, h)
        if key not in self._nadir_cache:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r = np.hypot((xx - cam.cx) / w, (yy - cam.cy) / h)
            self._nadir_cache[key] = (r / r.max() * 60000).astype(np.uint16)
        return self._nadir_cache[key]

    # -------------------------------------------------------------- placing
    def image_to_canvas_h(self, cam: CameraModel, pose, height_m, frame_id=None):
        w, h = cam.width, cam.height
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
        off, valid = cam.ground_offsets(src, pose.roll, pose.pitch, pose.yaw, height_m)
        if not valid.all():
            return None, None
        n0, e0 = self.frame.to_ne(pose.lat, pose.lon)
        gn, ge = n0 + off[:, 0], e0 + off[:, 1]
        return src, np.column_stack((gn, ge))

    def add(self, image: np.ndarray, cam: CameraModel, pose, height_m: float, frame_id: int) -> dict:
        """Add one photo. `cam` must describe `image` (same size)."""
        if cam.has_dist:
            image = cam.undistort_image(image)
            cam = cam.pinhole()
        src, ground = self.image_to_canvas_h(cam, pose, height_m)
        if src is None:
            return {"placed": False, "reason": "camera sees horizon"}
        fw, fl = cam.footprint(height_m)
        span = np.ptp(ground, axis=0)
        if max(span) > 4 * max(fw, fl):
            return {"placed": False, "reason": "footprint implausible"}
        if not self._ensure(ground[:, 0].min(), ground[:, 0].max(), ground[:, 1].min(), ground[:, 1].max()):
            return {"placed": False, "reason": "canvas full"}

        x, y = self.ne_to_px(ground[:, 0], ground[:, 1])
        dst = np.column_stack((x, y)).astype(np.float32)
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst).astype(np.float64)
        nadir = self._nadir_map(cam.width, cam.height, cam)

        warped, wscore, roi = self._warp(image, nadir, H)
        if warped is None:
            return {"placed": False, "reason": "outside canvas"}
        shift = (0.0, 0.0)
        resp = 0.0
        if self.refine and self.frames:
            est = self._estimate_shift(warped, wscore, roi)
            if est is not None:
                (dx, dy), resp = est
                if resp >= self.min_resp and math.hypot(dx, dy) <= self.max_shift_px:
                    # GPS anchoring: keep a fraction of the GPS placement so the mosaic tracks a
                    # running average of GPS positions instead of drifting away frame by frame.
                    k = 1.0 - self.gps_weight
                    shift = (dx * k, dy * k)
                    dx, dy = shift
                    H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]]) @ H
                    warped, wscore, roi = self._warp(image, nadir, H)
        x0, y0, x1, y1 = roi
        cv_roi = self.canvas[y0:y1, x0:x1]
        sc_roi = self.score[y0:y1, x0:x1]
        valid = wscore < EMPTY
        overlap = valid & (sc_roi < EMPTY)
        gain = 1.0
        if self.exposure_comp and overlap.sum() > 2000:
            a = cv_roi[overlap].astype(np.float32).mean()
            b = warped[overlap].astype(np.float32).mean()
            if b > 5:
                gain = float(np.clip(a / b, 0.8, 1.25))
                if abs(gain - 1) > 0.01:
                    warped = np.clip(warped.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        upd = valid & (wscore < sc_roi)
        cv_roi[upd] = warped[upd]
        sc_roi[upd] = wscore[upd]
        corr_n, corr_e = -shift[1] * self.res, shift[0] * self.res
        info = {"placed": True, "frame_id": frame_id, "H": H.tolist(), "shift_m": [corr_n, corr_e],
                "refine_response": round(float(resp), 3), "gain": round(gain, 3),
                "overlap_frac": round(float(overlap.sum()) / max(1, valid.sum()), 3),
                "img_size": [cam.width, cam.height]}
        self.frames[frame_id] = info
        return info

    def _warp(self, image, nadir, H):
        h, w = self.shape
        corners = cv2.perspectiveTransform(
            np.array([[[0, 0]], [[image.shape[1], 0]], [[image.shape[1], image.shape[0]]], [[0, image.shape[0]]]],
                     np.float32), H.astype(np.float32)).reshape(-1, 2)
        x0, y0 = np.floor(corners.min(0)).astype(int)
        x1, y1 = np.ceil(corners.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None, None, None
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64) @ H
        size = (x1 - x0, y1 - y0)
        warped = cv2.warpPerspective(image, T, size, flags=cv2.INTER_AREA if self._downscale(H) else cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        wscore = cv2.warpPerspective(nadir, T, size, flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=int(EMPTY))
        return warped, wscore, (x0, y0, x1, y1)

    @staticmethod
    def _downscale(H) -> bool:
        return abs(np.linalg.det(H[:2, :2])) < 1.0

    def _estimate_shift(self, warped, wscore, roi):
        """Residual (dx, dy) in canvas px that best aligns warped onto the existing map."""
        x0, y0, x1, y1 = roi
        existing = self.score[y0:y1, x0:x1] < EMPTY
        overlap = existing & (wscore < EMPTY)
        if overlap.mean() < 0.12 or overlap.sum() < 4000:
            return None
        ys, xs = np.nonzero(overlap)
        by0, by1, bx0, bx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if (by1 - by0) < 64 or (bx1 - bx0) < 64:
            return None
        a = cv2.cvtColor(warped[by0:by1, bx0:bx1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        b = cv2.cvtColor(self.canvas[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = overlap[by0:by1, bx0:bx1]
        f = 1.0
        if max(a.shape) > 512:                       # speed: correlate at <= 512 px
            f = 512.0 / max(a.shape)
            a = cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            m = cv2.resize(m.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        # high-pass so the match uses texture, not illumination; zero outside overlap
        a = a - cv2.GaussianBlur(a, (0, 0), 6)
        b = b - cv2.GaussianBlur(b, (0, 0), 6)
        a[~m] = 0
        b[~m] = 0
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
        return (dx / f, dy / f), resp

    # --------------------------------------------------------------- output
    def valid_bbox(self, margin_px: int = 20):
        if self.score is None:
            return None
        rows = np.where((self.score < EMPTY).any(1))[0]
        cols = np.where((self.score < EMPTY).any(0))[0]
        if len(rows) == 0:
            return None
        h, w = self.shape
        return (max(0, cols[0] - margin_px), max(0, rows[0] - margin_px),
                min(w, cols[-1] + margin_px), min(h, rows[-1] + margin_px))

    def coverage_m2(self) -> float:
        return 0.0 if self.score is None else float((self.score < EMPTY).sum()) * self.res ** 2

    def preview(self, max_side: int = 1600) -> np.ndarray | None:
        bb = self.valid_bbox()
        if bb is None:
            return None
        x0, y0, x1, y1 = bb
        img = self.canvas[y0:y1, x0:x1]
        f = min(1.0, max_side / max(img.shape[:2]))
        return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img.copy()
