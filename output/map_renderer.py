"""
Draws the final competition map: mosaic + tent bounding box + GPS label +
close-up inset + north arrow + scale bar + information panel.
"""
from __future__ import annotations

import datetime as dt
import math

import cv2
import numpy as np

from mapping.gps_converter import to_dms_ascii
from mapping.image_mapper import GeoMosaic

FONT = cv2.FONT_HERSHEY_SIMPLEX
RED = (30, 30, 230)
YELLOW = (0, 230, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _text(img, txt, org, scale, color=WHITE, thick=1, bg=None):
    (tw, th), base = cv2.getTextSize(txt, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + base + 2), bg, -1)
    cv2.putText(img, txt, (x, y), FONT, scale, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return tw, th


def _nice_length(target_m: float) -> float:
    exp = 10 ** math.floor(math.log10(target_m))
    for m in (1, 2, 5, 10):
        if m * exp >= target_m:
            return m * exp
    return 10 * exp


def render_map(mosaic: GeoMosaic, estimate, stats: dict, team: str, closeup: np.ndarray | None = None,
               max_px: int = 12000) -> tuple[np.ndarray, dict]:
    bb = mosaic.valid_bbox(margin_px=int(5 / mosaic.res))
    if bb is None:
        raise RuntimeError("mosaic is empty")
    x0, y0, x1, y1 = bb
    img = mosaic.canvas[y0:y1, x0:x1].copy()
    f = min(1.0, max_px / max(img.shape[:2]))
    if f < 1:
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    res = mosaic.res / f                       # metres per output pixel
    H, W = img.shape[:2]
    s = max(0.6, W / 2400)                     # text scale relative to map size
    t = max(2, int(round(2 * s)))
    geo = {"x0": x0, "y0": y0, "scale": f, "res_m": res}

    def to_img(n, e):
        x, y = mosaic.ne_to_px(n, e)
        return (np.asarray(x) - x0) * f, (np.asarray(y) - y0) * f

    # ------------------------------------------------------------ tent box
    if estimate is not None:
        best = estimate.best
        c = np.array(best.corners_ne)
        c = c - c.mean(0) + [estimate.map_n, estimate.map_e]        # shape of best view at fused position
        px, py = to_img(c[:, 0], c[:, 1])
        poly = np.column_stack((px, py)).astype(np.int32)
        pad = max(6, int(0.6 / res))
        bx0, by0 = poly.min(0) - pad
        bx1, by1 = poly.max(0) + pad
        cv2.polylines(img, [poly], True, YELLOW, max(1, t // 2), cv2.LINE_AA)
        cv2.rectangle(img, (int(bx0), int(by0)), (int(bx1), int(by1)), RED, t + 1, cv2.LINE_AA)
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        cv2.drawMarker(img, (int(cx), int(cy)), RED, cv2.MARKER_CROSS, int(pad * 1.2), max(1, t // 2))
        lines = ["WHITE TENT", f"LAT {estimate.lat:.7f}", f"LON {estimate.lon:.7f}",
                 f"conf {estimate.confidence:.2f}  +/-{estimate.error_95_m:.1f} m (95%)"]
        lx, ly = int(bx1 + 12 * s), int(by0 + 10 * s)
        widths = [cv2.getTextSize(l, FONT, 0.75 * s, t)[0][0] for l in lines]
        if lx + max(widths) + 20 > W:
            lx = int(bx0 - max(widths) - 20 * s)
        lh = int(32 * s)
        ly = int(np.clip(ly, lh, H - lh * len(lines) - 10))
        cv2.rectangle(img, (lx - 8, ly - lh), (lx + max(widths) + 8, ly + lh * (len(lines) - 1) + 12), (40, 40, 40), -1)
        cv2.rectangle(img, (lx - 8, ly - lh), (lx + max(widths) + 8, ly + lh * (len(lines) - 1) + 12), RED, max(1, t // 2))
        for i, l in enumerate(lines):
            _text(img, l, (lx, ly + i * lh), 0.75 * s, YELLOW if i == 0 else WHITE, t)
        geo["tent_px"] = [float(cx), float(cy)]
        geo["tent_box_px"] = [int(bx0), int(by0), int(bx1), int(by1)]

        # close-up inset, top-right
        if closeup is not None and closeup.size:
            iw = int(min(W, H) * 0.28)
            ins = cv2.resize(closeup, (iw, int(iw * closeup.shape[0] / closeup.shape[1])), interpolation=cv2.INTER_AREA)
            ih = ins.shape[0]
            ox, oy = W - iw - int(20 * s), int(20 * s)
            if oy + ih < H:
                img[oy:oy + ih, ox:ox + iw] = ins
                cv2.rectangle(img, (ox, oy), (ox + iw, oy + ih), RED, t)
                _text(img, "tent close-up (onboard camera)", (ox + 6, oy + ih + int(26 * s)), 0.55 * s, WHITE, max(1, t - 1))

    # ------------------------------------------------------- north arrow
    ax, ay, al = int(70 * s), int(50 * s), int(90 * s)
    pts = np.array([[ax, ay], [ax - al // 4, ay + al], [ax, ay + int(al * 0.78)], [ax + al // 4, ay + al]], np.int32)
    cv2.fillPoly(img, [pts], WHITE, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, BLACK, max(1, t // 2), cv2.LINE_AA)
    _text(img, "N", (ax - int(12 * s), ay + al + int(40 * s)), 1.1 * s, WHITE, t + 1)

    # --------------------------------------------------------- scale bar
    length_m = _nice_length(W * res * 0.18)
    bar = int(length_m / res)
    sx, sy = int(30 * s), H - int(40 * s)
    seg = bar // 4
    cv2.rectangle(img, (sx - int(14 * s), sy - int(52 * s)), (sx + bar + int(14 * s), sy + int(14 * s)), (60, 60, 60), -1)
    for i in range(4):
        cv2.rectangle(img, (sx + i * seg, sy - int(12 * s)), (sx + (i + 1) * seg, sy), WHITE if i % 2 == 0 else BLACK, -1)
    cv2.rectangle(img, (sx, sy - int(12 * s)), (sx + bar, sy), BLACK, max(1, t // 2))
    _text(img, "0", (sx - 5, sy - int(20 * s)), 0.6 * s, WHITE, max(1, t - 1))
    _text(img, f"{length_m:g} m", (sx + bar - int(30 * s), sy - int(20 * s)), 0.6 * s, WHITE, max(1, t - 1))

    # ------------------------------------------------------------ panel
    ph = int(200 * s)
    panel = np.full((ph, W, 3), 32, np.uint8)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    left = [f"{team}  |  Risk map & target report", f"Generated onboard (Raspberry Pi) {now}",
            f"Photos: {stats.get('images', 0)}   Area: {stats.get('area_m2', 0):,.0f} m2   "
            f"Map resolution: {res * 100:.1f} cm/px   North up",
            f"Detector: {stats.get('detector', '?')}   Map: GPS placement + image alignment"]
    if estimate is not None:
        right = ["TENT GPS (WGS-84)", f"{estimate.lat:.7f}, {estimate.lon:.7f}",
                 f"{to_dms_ascii(estimate.lat, True)}  {to_dms_ascii(estimate.lon, False)}",
                 f"Conf {estimate.confidence:.2f} | {estimate.sightings} sightings | +/-{estimate.error_95_m:.1f} m 95%"]
    else:
        right = ["TENT NOT DETECTED", "", "", ""]
    lh = int(42 * s)
    for i, l in enumerate(left):
        _text(panel, l, (int(20 * s), int(48 * s) + i * lh), (0.75 if i == 0 else 0.6) * s, YELLOW if i == 0 else WHITE, t if i == 0 else max(1, t - 1))
    rx = int(W * 0.56)
    for i, l in enumerate(right):
        _text(panel, l, (rx, int(48 * s) + i * lh), (0.8 if i < 2 else 0.6) * s, RED if i == 0 else WHITE, t if i < 2 else max(1, t - 1))
    out = np.vstack((img, panel))
    return out, geo


def make_closeup(image: np.ndarray, bbox, conf: float, context: float = 3.0) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * context / 2
    h, w = image.shape[:2]
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(w, cx + half)), int(min(h, cy + half))
    crop = image[b:d, a:c].copy()
    t = max(2, crop.shape[1] // 150)
    cv2.rectangle(crop, (int(x1 - a), int(y1 - b)), (int(x2 - a), int(y2 - b)), RED, t)
    cv2.putText(crop, f"white_tent {conf:.2f}", (int(x1 - a), max(15, int(y1 - b) - 6)), FONT,
                max(0.4, crop.shape[1] / 600), YELLOW, max(1, t // 2), cv2.LINE_AA)
    return crop
