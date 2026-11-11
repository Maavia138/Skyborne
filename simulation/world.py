"""
Synthetic aerial world for testing the whole pipeline without a drone.

A large north-up ground texture (default 240 m x 220 m at 5 cm/px) with
grass, dry patches, a dirt track, trees with shadows and - most importantly -
the white tent plus HARD NEGATIVES that a naive white detector would flag:
    * white car (dark windows inside)          * white-roof building (too big)
    * concrete pad (bright, low saturation)    * white road markings (thin)
    * white stones (tiny)                      * orange tent (right size, wrong colour)
Ground-truth positions are known, so we can measure detection and GPS error.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from config.settings import SimConfig
from mapping.camera_model import CameraModel
from mapping.gps_converter import LocalFrame

SHIFT = 3            # fixed-point bits for anti-aliased polygons
SCALE = 1 << SHIFT


def rect_ne(n, e, length, width, heading_rad):
    """Corners (N,E) of a rectangle centred at (n,e), long axis along heading."""
    u = np.array([math.cos(heading_rad), math.sin(heading_rad)])
    v = np.array([-math.sin(heading_rad), math.cos(heading_rad)])
    c = np.array([n, e])
    hl, hw = length / 2, width / 2
    return np.array([c + hl * u + hw * v, c + hl * u - hw * v, c - hl * u - hw * v, c - hl * u + hw * v])


class SyntheticWorld:
    def __init__(self, cfg: SimConfig, seed: int | None = None, center_latlon=None, place_tent: bool = True,
                 tent_style: str = "ridge"):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed if seed is None else seed)
        self.rng = rng
        self.res = cfg.world_res_m
        self.size_n, self.size_e = cfg.world_size_m
        lat0, lon0 = center_latlon or (cfg.home_lat, cfg.home_lon)
        self.frame = LocalFrame(lat0, lon0)
        self.H = int(self.size_n / self.res)
        self.W = int(self.size_e / self.res)
        self.sun = rng.uniform(0, 2 * math.pi)             # shadow direction
        self.objects: list[dict] = []
        self.tex = self._ground(rng)
        self._track(rng)
        self._vegetation(rng)
        if cfg.decoys:
            self._decoys(rng)
        self.tent = None
        if place_tent:
            sn, se = cfg.survey_size_m
            if cfg.tent_ne is not None:
                tn, te = cfg.tent_ne
            else:
                for _ in range(200):
                    tn, te = rng.uniform(-0.4, 0.4) * sn, rng.uniform(-0.4, 0.4) * se
                    if all(math.hypot(tn - o["n"], te - o["e"]) > 12 for o in self.objects):
                        break
            L, Wd = cfg.tent_size_m
            self.tent = self.draw_tent(tn, te, L, Wd, rng.uniform(0, math.pi), rng, tent_style)

    # ------------------------------------------------------------ helpers
    def ne_to_tex(self, n, e):
        return (np.asarray(e) + self.size_e / 2) / self.res, (self.size_n / 2 - np.asarray(n)) / self.res

    def _px(self, pts_ne):
        x, y = self.ne_to_tex(pts_ne[:, 0], pts_ne[:, 1])
        return np.round(np.column_stack((x, y)) * SCALE).astype(np.int32)

    def fill(self, pts_ne, color):
        cv2.fillPoly(self.tex, [self._px(pts_ne)], tuple(int(c) for c in color), cv2.LINE_AA, SHIFT)

    def shade(self, pts_ne, factor):
        """Multiply texture inside polygon by factor (shadows)."""
        px = self._px(pts_ne) / SCALE
        x0, y0 = np.floor(px.min(0)).astype(int) - 2
        x1, y1 = np.ceil(px.max(0)).astype(int) + 2
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.W, x1), min(self.H, y1)
        if x1 <= x0 or y1 <= y0:
            return
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [np.round((px - [x0, y0]) * SCALE).astype(np.int32)], 255, cv2.LINE_AA, SHIFT)
        m = cv2.GaussianBlur(m, (0, 0), 1.5).astype(np.float32) / 255.0
        roi = self.tex[y0:y1, x0:x1].astype(np.float32)
        roi *= (1 - m * (1 - factor))[..., None]
        self.tex[y0:y1, x0:x1] = roi.astype(np.uint8)

    def shadow_of(self, pts_ne, height_m, factor=0.55):
        d = 0.8 * height_m * np.array([math.cos(self.sun), math.sin(self.sun)])
        hull = cv2.convexHull(np.vstack((pts_ne, pts_ne + d)).astype(np.float32)).reshape(-1, 2)
        self.shade(hull.astype(np.float64), factor)

    def circle_pts(self, n, e, r, k=24):
        a = np.linspace(0, 2 * math.pi, k, endpoint=False)
        return np.column_stack((n + r * np.cos(a), e + r * np.sin(a)))

    # ------------------------------------------------------------- ground
    def _noise(self, rng, cell_m):
        gh = max(2, int(self.H * self.res / cell_m)) + 3
        gw = max(2, int(self.W * self.res / cell_m)) + 3
        return cv2.resize(rng.random((gh, gw), dtype=np.float32), (self.W, self.H), interpolation=cv2.INTER_CUBIC)

    def _ground(self, rng):
        base = self._noise(rng, 25.0)
        base *= 0.45
        base += 0.35 * self._noise(rng, 5.0)
        base += 0.20 * self._noise(rng, 0.8)
        dry = self._noise(rng, 40.0)
        np.clip((dry - 0.55) * 3.0, 0, 1, out=dry)
        palette = rng.choice([0, 1, 2])
        dark = [(38, 88, 52), (40, 95, 75), (45, 85, 70)][palette]
        light = [(62, 140, 98), (70, 150, 125), (80, 135, 120)][palette]
        dryc = (92, 150, 170)
        grain = rng.integers(-10, 11, (self.H, self.W), dtype=np.int16).astype(np.float32)
        tex = np.empty((self.H, self.W, 3), np.uint8)
        for c in range(3):
            ch = dark[c] + (light[c] - dark[c]) * base
            ch = ch * (1 - dry) + dryc[c] * dry + grain
            tex[..., c] = np.clip(ch, 0, 255).astype(np.uint8)
        return tex

    def _track(self, rng):
        n_pts = 6
        ns = np.linspace(-self.size_n / 2, self.size_n / 2, n_pts)
        es = rng.uniform(-0.35, 0.35) * self.size_e + np.cumsum(rng.normal(0, 8, n_pts))
        pts = self._px(np.column_stack((ns, es)))
        mask = np.zeros((self.H, self.W), np.uint8)
        cv2.polylines(mask, [pts], False, 255, int(2.8 / self.res), cv2.LINE_AA, SHIFT)
        mask = cv2.GaussianBlur(mask, (0, 0), 4)
        m = mask.astype(np.float32)[..., None] / 255.0
        soil = np.array([88, 118, 140], np.float32)
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        roi = self.tex[y0:y1, x0:x1].astype(np.float32)
        roi = roi * (1 - m[y0:y1, x0:x1]) + (soil + rng.normal(0, 6, roi.shape)) * m[y0:y1, x0:x1]
        self.tex[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
        self.track_ne = np.column_stack((ns, es))

    def _vegetation(self, rng):
        for _ in range(int(self.size_n * self.size_e / 1500)):
            n, e = rng.uniform(-0.48, 0.48) * self.size_n, rng.uniform(-0.48, 0.48) * self.size_e
            r = rng.uniform(1.5, 4.0)
            self.shadow_of(self.circle_pts(n, e, r), r * 1.6, 0.5)
            g = rng.integers(0, 20)
            for k, f in enumerate((1.0, 0.75, 0.45)):
                col = (25 + g + 8 * k, 62 + g + 14 * k, 30 + g + 8 * k)
                self.fill(self.circle_pts(n + 0.15 * r * k, e - 0.15 * r * k, r * f, 20), col)
        for _ in range(int(self.size_n * self.size_e / 500)):
            n, e = rng.uniform(-0.49, 0.49) * self.size_n, rng.uniform(-0.49, 0.49) * self.size_e
            r = rng.uniform(0.4, 1.1)
            self.shadow_of(self.circle_pts(n, e, r, 12), r, 0.6)
            self.fill(self.circle_pts(n, e, r, 12), (35, 80 + int(rng.integers(0, 25)), 45))

    def _free_spot(self, rng, min_d=14):
        for _ in range(300):
            n, e = rng.uniform(-0.42, 0.42) * self.size_n, rng.uniform(-0.42, 0.42) * self.size_e
            if all(math.hypot(n - o["n"], e - o["e"]) > min_d for o in self.objects):
                return n, e
        return n, e

    def _decoys(self, rng):
        sn, se = self.cfg.survey_size_m
        inside = lambda: (rng.uniform(-0.45, 0.45) * sn, rng.uniform(-0.45, 0.45) * se)  # noqa: E731
        # white car
        n, e = inside()
        h = rng.uniform(0, math.pi)
        body = rect_ne(n, e, 4.5, 1.8, h)
        self.shadow_of(body, 1.4)
        self.fill(body, (236, 238, 238))
        u = np.array([math.cos(h), math.sin(h)])
        self.fill(rect_ne(n + 0.9 * u[0], e + 0.9 * u[1], 0.8, 1.5, h), (45, 45, 50))     # windscreen
        self.fill(rect_ne(n - 1.3 * u[0], e - 1.3 * u[1], 0.5, 1.4, h), (45, 45, 50))     # rear window
        self.fill(rect_ne(n - 0.2 * u[0], e - 0.2 * u[1], 1.6, 1.3, h), (225, 228, 228))   # roof
        self.objects.append({"type": "white_car", "n": n, "e": e, "corners": body.tolist()})
        # white-roof building
        n, e = self._free_spot(rng)
        h = rng.uniform(0, math.pi)
        roof = rect_ne(n, e, 12, 8, h)
        self.shadow_of(roof, 4.0)
        self.fill(roof, (228, 230, 232))
        self.fill(rect_ne(n, e, 12, 0.25, h), (190, 192, 195))
        self.objects.append({"type": "white_building", "n": n, "e": e, "corners": roof.tolist()})
        # concrete pad
        n, e = self._free_spot(rng)
        pad = rect_ne(n, e, 7, 6, rng.uniform(0, math.pi))
        self.fill(pad, (168, 172, 172))
        self.objects.append({"type": "concrete_pad", "n": n, "e": e, "corners": pad.tolist()})
        # road markings along the track
        for k in range(6):
            i = rng.integers(0, len(self.track_ne) - 1)
            a, b = self.track_ne[i], self.track_ne[i + 1]
            p = a + (b - a) * rng.uniform(0.1, 0.9)
            hd = math.atan2(b[1] - a[1], b[0] - a[0])
            self.fill(rect_ne(p[0], p[1], 3.0, 0.15, hd), (240, 240, 240))
        # white stones
        for _ in range(25):
            n, e = inside()
            self.fill(self.circle_pts(n, e, rng.uniform(0.12, 0.3), 10), (225, 228, 228))
        # orange tent (right size, wrong colour)
        n, e = self._free_spot(rng)
        self.draw_tent(n, e, 2.3, 2.0, rng.uniform(0, math.pi), rng, "ridge", color=((40, 120, 235), (30, 95, 200)),
                       kind="orange_tent")

    # --------------------------------------------------------------- tent
    def draw_tent(self, n, e, L, Wd, heading, rng, style="ridge", color=None, kind="white_tent") -> dict:
        sunlit, shaded = color or ((246, 246, 244), (214, 216, 216))
        foot = rect_ne(n, e, L, Wd, heading)
        # guy ropes + pegs
        u = np.array([math.cos(heading), math.sin(heading)])
        v = np.array([-math.sin(heading), math.cos(heading)])
        c = np.array([n, e])
        for su in (-1, 1):
            for sv in (-1, 1):
                a = c + su * L / 2 * u + sv * Wd / 2 * v
                b = a + su * 0.5 * u + sv * 0.6 * v
                d = b - a
                self.fill(rect_ne(*((a + b) / 2), np.linalg.norm(d), 0.03, math.atan2(d[1], d[0])), (150, 150, 150))
        self.shadow_of(foot, 1.3, 0.5)
        if style == "dome":
            pts = self.circle_pts(n, e, Wd / 2, 10)
            self.fill(pts, sunlit)
            half = np.vstack((c, pts[:6]))
            self.fill(half, shaded)
            for ang in (heading + math.pi / 4, heading - math.pi / 4):
                self.fill(rect_ne(n, e, Wd, 0.05, ang), (185, 188, 190))
        else:
            # two roof halves split along the ridge (sun side brighter)
            side_a = np.array([c + L / 2 * u, c + L / 2 * u + Wd / 2 * v, c - L / 2 * u + Wd / 2 * v, c - L / 2 * u])
            side_b = np.array([c + L / 2 * u, c + L / 2 * u - Wd / 2 * v, c - L / 2 * u - Wd / 2 * v, c - L / 2 * u])
            lit_a = math.cos(self.sun - math.atan2(v[1], v[0])) < 0
            self.fill(side_a, sunlit if lit_a else shaded)
            self.fill(side_b, shaded if lit_a else sunlit)
            self.fill(rect_ne(n, e, L, 0.06, heading), (180, 182, 184))
            # door panel at one end
            d = c + (L / 2 - 0.25) * u
            self.fill(rect_ne(d[0], d[1], 0.4, Wd * 0.5, heading), tuple(int(x * 0.88) for x in shaded))
        obj = {"type": kind, "n": float(n), "e": float(e), "corners": foot.tolist(), "size_m": [L, Wd],
               "heading_deg": math.degrees(heading)}
        self.objects.append(obj)
        lat, lon = self.frame.to_latlon(n, e)
        obj.update(lat=lat, lon=lon)
        return obj

    # -------------------------------------------------------------- render
    def survey_polygon_latlon(self):
        sn, se = self.cfg.survey_size_m
        corners = [(-sn / 2, -se / 2), (sn / 2, -se / 2), (sn / 2, se / 2), (-sn / 2, se / 2)]
        return [list(self.frame.to_latlon(n, e)) for n, e in corners]

    def render(self, pose, cam: CameraModel, height_m: float, rng=None, effects: bool = True) -> np.ndarray:
        """Render what a camera at `pose` (true pose, flat ground) would see."""
        w, h = cam.width, cam.height
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
        off, valid = cam.ground_offsets(corners, pose.roll, pose.pitch, pose.yaw, height_m)
        if not valid.all():
            return np.zeros((h, w, 3), np.uint8)
        n0, e0 = self.frame.to_ne(pose.lat, pose.lon)
        tx, ty = self.ne_to_tex(n0 + off[:, 0], e0 + off[:, 1])
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), np.column_stack((tx, ty)).astype(np.float32))
        img = cv2.warpPerspective(self.tex, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REFLECT_101)
        if not effects:
            return img
        rng = rng or self.rng
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        gain = (1 - 0.12 * r2) * rng.uniform(0.92, 1.08)
        out = img.astype(np.float32) * gain[..., None] + rng.normal(0, 2.5, img.shape).astype(np.float32)
        out = cv2.GaussianBlur(out, (0, 0), 0.6)
        return np.clip(out, 0, 255).astype(np.uint8)

    def project_object(self, obj, pose, cam: CameraModel, height_m: float):
        """Image bbox of a world object (for dataset labels) or None if not visible."""
        n0, e0 = self.frame.to_ne(pose.lat, pose.lon)
        c = np.array(obj["corners"]) - [n0, e0]
        uv, ok = cam.project_offsets(c, pose.roll, pose.pitch, pose.yaw, height_m)
        if not ok.all():
            return None
        x1, y1 = uv.min(0)
        x2, y2 = uv.max(0)
        return [float(x1), float(y1), float(x2), float(y2)]
