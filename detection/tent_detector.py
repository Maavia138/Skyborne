"""
White tent detection.

Stage 1 - WhiteBlobDetector (always on, ~30 ms on a Pi 5 at 1640 px)
    1. HSV: "white" = low saturation AND bright. The brightness threshold adapts
       to each photo (sun, clouds, auto-gain), so it is not a fixed number.
    2. Top-hat: the object must be brighter than its surroundings (rejects
       large bright areas such as dry ground or concrete).
    3. Contours + REAL-WORLD SIZE. Because we know the altitude, every pixel
       has a size in metres (GSD). A tent is ~2-20 m^2 and 1.2-6 m per side.
       That one test removes white roofs, road lines, stones and paper.
    4. Shape: solidity, aspect ratio, and "dark pixels inside" (white cars have
       dark windows inside the white body; tents do not).
    Each test gives a 0..1 score; the confidence is their geometric mean.

Stage 2 - YOLO (optional, detection/yolo_detector.py)
    A YOLO11n / YOLOv8n model fine-tuned on aerial tent images, exported to
    ONNX and run on the Pi CPU. When the model file exists, both stages run
    and the results are fused: a YOLO box confirmed by the blob detector gets
    a higher confidence; a blob without YOLO support gets a lower one.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from config.settings import DetectionConfig

log = logging.getLogger("detector")


@dataclass
class Detection:
    bbox: list                       # [x_min, y_min, x_max, y_max] in full-resolution pixels
    confidence: float
    source: str = "hsv"              # hsv | yolo | fused
    cls: str = "white_tent"
    truncated: bool = False          # touches the image border
    area_m2: float = 0.0
    size_m: list = field(default_factory=lambda: [0.0, 0.0])
    scores: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = [round(float(v), 1) for v in self.bbox]
        d["confidence"] = round(float(self.confidence), 3)
        d["detected"] = True
        return d


def _trap(x, a, b, c, d):
    """Trapezoid membership: 0 below a, ramps to 1 at b, 1 until c, 0 at d."""
    if x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    return (d - x) / (d - c)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class WhiteBlobDetector:
    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg

    def white_mask(self, img: np.ndarray, gsd: float) -> tuple[np.ndarray, int, np.ndarray]:
        """gsd = metres per pixel of `img`."""
        c = self.cfg
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s, v = hsv[..., 1], hsv[..., 2]
        med = float(np.median(v[::4, ::4]))
        p999 = float(np.percentile(v[::2, ::2], 99.9))
        thr = int(np.clip(med + c.adaptive_v_frac * (p999 - med), c.white_v_min, 250))
        # Local background: grey-level opening with a kernel larger than the biggest tent,
        # computed at 1/8 scale (fast). Anything tent-sized and bright stands out above it.
        f = 8
        small = cv2.resize(v, (max(8, v.shape[1] // f), max(8, v.shape[0] // f)), interpolation=cv2.INTER_AREA)
        k = int(1.4 * c.tent_max_side_m / (gsd * f)) | 1
        bg = cv2.morphologyEx(small, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        bg = cv2.blur(bg, (max(3, k // 2), max(3, k // 2)))
        bg = cv2.resize(bg, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR)
        contrast = cv2.subtract(v, bg)
        mask = ((s <= c.white_s_max) & (v >= thr) & (contrast >= c.tophat_min)).astype(np.uint8) * 255
        return mask, thr, hsv

    def detect(self, img_bgr: np.ndarray, gsd_full_m: float) -> list[Detection]:
        """gsd_full_m: metres per pixel of the full-resolution image."""
        c = self.cfg
        H, W = img_bgr.shape[:2]
        scale = min(1.0, c.proc_width / W)
        img = cv2.resize(img_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
        gsd = gsd_full_m / scale
        h, w = img.shape[:2]
        mask, thr, hsv = self.white_mask(img, gsd)

        # close small gaps (ridge lines, seams) but not more than ~0.4 m
        kc = max(3, int(round(0.4 / gsd)) | 1)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        min_px = 0.5 * c.tent_min_area_m2 / (gsd * gsd)
        max_px = 2.0 * c.tent_max_area_m2 / (gsd * gsd)
        dets = []
        for i in range(1, n):
            area_px = stats[i, cv2.CC_STAT_AREA]
            if area_px < min_px or area_px > max_px:
                continue
            x, y, bw, bh = stats[i, :4]
            comp = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnts, key=cv2.contourArea)
            (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
            long_m, short_m = max(rw, rh) * gsd, max(min(rw, rh), 1) * gsd
            area_m2 = area_px * gsd * gsd
            hull_area = cv2.contourArea(cv2.convexHull(cnt)) or 1.0
            solidity = min(1.0, area_px / hull_area)
            aspect = long_m / short_m

            roi_hsv = hsv[y:y + bh, x:x + bw]
            on = comp.astype(bool)
            mean_s = float(roi_hsv[..., 1][on].mean())
            mean_v = float(roi_hsv[..., 2][on].mean())
            # dark pixels inside the component's filled hull (car windows, solar panels)
            hull_mask = np.zeros_like(comp)
            cv2.drawContours(hull_mask, [cv2.convexHull(cnt)], -1, 1, -1)
            inside = hull_mask.astype(bool)
            dark_inside = float((roi_hsv[..., 2][inside] < 90).mean()) if inside.any() else 0.0

            sc = {
                "area": _trap(area_m2, c.tent_min_area_m2 * 0.6, c.tent_min_area_m2, c.tent_max_area_m2, c.tent_max_area_m2 * 1.5),
                "side": min(_trap(long_m, c.tent_min_side_m * 0.7, c.tent_min_side_m, c.tent_max_side_m, c.tent_max_side_m * 1.3),
                            _trap(short_m, c.tent_min_side_m * 0.6, c.tent_min_side_m * 0.9, 99, 100)),
                "aspect": _trap(aspect, 0.0, 0.5, c.tent_max_aspect * 0.8, c.tent_max_aspect * 1.25),
                "solidity": float(np.clip((solidity - 0.68) / 0.2, 0, 1)),
                "white": float(np.clip((mean_v - thr + 40) / 50, 0, 1) * np.clip((c.white_s_max + 10 - mean_s) / 30, 0, 1)),
                "no_dark": float(np.clip(1 - dark_inside / c.max_dark_inside, 0, 1)),
            }
            vals = list(sc.values())
            conf = 0.0 if min(vals) <= 0 else float(np.exp(np.mean(np.log(vals))))
            truncated = bool(x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1)
            if truncated:
                conf *= 0.6
            if conf < c.min_confidence:
                continue
            pad = 0.15 / gsd  # small margin around the tent fabric
            bbox = [max(0, x - pad) / scale, max(0, y - pad) / scale,
                    min(w, x + bw + pad) / scale, min(h, y + bh + pad) / scale]
            dets.append(Detection(bbox=bbox, confidence=conf, source="hsv", truncated=truncated,
                                  area_m2=round(area_m2, 2), size_m=[round(long_m, 2), round(short_m, 2)],
                                  scores={k: round(v, 2) for k, v in sc.items()}))
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets


class HybridTentDetector:
    """Blob detector + optional YOLO, fused."""

    def __init__(self, cfg: DetectionConfig, model_path: str | Path | None = None):
        self.cfg = cfg
        self.blob = WhiteBlobDetector(cfg)
        self.yolo = None
        if cfg.use_yolo != "off":
            path = Path(model_path or cfg.yolo_model)
            if path.exists():
                from detection.yolo_detector import YoloDetector
                self.yolo = YoloDetector(path, cfg)
                log.info("YOLO model loaded: %s (%s)", path, self.yolo.backend)
            elif cfg.use_yolo == "on":
                raise FileNotFoundError(f"YOLO model not found: {path}")
            else:
                log.info("No YOLO model at %s - using white-blob detector only", path)

    @property
    def mode(self) -> str:
        return "yolo+hsv" if self.yolo else "hsv"

    def detect(self, img_bgr: np.ndarray, gsd_full_m: float) -> list[Detection]:
        blobs = self.blob.detect(img_bgr, gsd_full_m)
        if not self.yolo:
            return blobs
        c = self.cfg
        out = []
        used = set()
        for y in self.yolo.detect(img_bgr):
            # real-world size gate for neural boxes too
            w_m = (y.bbox[2] - y.bbox[0]) * gsd_full_m
            h_m = (y.bbox[3] - y.bbox[1]) * gsd_full_m
            if max(w_m, h_m) > c.tent_max_side_m * 1.6 or min(w_m, h_m) < c.tent_min_side_m * 0.5:
                continue
            best, best_iou = None, 0.0
            for j, b in enumerate(blobs):
                o = iou(y.bbox, b.bbox)
                if o > best_iou:
                    best, best_iou = j, o
            if best is not None and best_iou > 0.3:
                used.add(best)
                b = blobs[best]
                y.confidence = 1 - (1 - y.confidence) * (1 - 0.5 * b.confidence)
                y.source, y.area_m2, y.size_m, y.scores = "fused", b.area_m2, b.size_m, b.scores
            else:
                y.confidence *= 0.9
                y.size_m = [round(max(w_m, h_m), 2), round(min(w_m, h_m), 2)]
            out.append(y)
        for j, b in enumerate(blobs):
            if j not in used:
                b.confidence *= c.hsv_only_weight
                if b.confidence >= c.min_confidence * c.hsv_only_weight:
                    out.append(b)
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out


def draw_detections(img: np.ndarray, dets: list[Detection]) -> np.ndarray:
    out = img.copy()
    t = max(2, img.shape[1] // 500)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        col = (0, 0, 255) if d.confidence >= 0.6 else (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, t)
        cv2.putText(out, f"{d.cls} {d.confidence:.2f} [{d.source}]", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, col, t, cv2.LINE_AA)
    return out
