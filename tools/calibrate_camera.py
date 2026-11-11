"""
Camera calibration with a printed chessboard (do this once with the real Pi Camera v2).

1. Print a chessboard (e.g. 9x6 inner corners, 25 mm squares), tape it flat to a board.
2. On the Pi, take 20-30 full-resolution photos from different angles/distances:
       python tools/calibrate_camera.py capture --n 25
3. Compute intrinsics + distortion:
       python tools/calibrate_camera.py solve --images data/calib --cols 9 --rows 6 --square 0.025
   -> config/camera_calibration.json (loaded automatically by config/settings.py)

A good result has RMS reprojection error < 0.5 px. Calibration improves tent
GPS accuracy towards the image edges (the Pi Camera v2 lens has noticeable
barrel distortion).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def capture(a):
    from camera.camera_capture import Picamera2Camera
    from config.settings import Settings
    cfg = Settings.load()
    cfg.camera.exposure_us = 10000
    cam = Picamera2Camera(cfg.camera).start()
    out = ROOT / "data" / "calib"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        input(f"[{i + 1}/{a.n}] position the board and press Enter ...")
        frame, _ = cam.capture()
        cv2.imwrite(str(out / f"calib_{i:02d}.png"), frame)
    cam.stop()


def solve(a):
    pattern = (a.cols, a.rows)
    objp = np.zeros((a.cols * a.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
    objpts, imgpts, size = [], [], None
    for p in sorted(Path(a.images).glob("*.*")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        size = img.shape[::-1]
        ok, corners = cv2.findChessboardCorners(img, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            print("no board:", p.name)
            continue
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
        objpts.append(objp)
        imgpts.append(corners)
    if len(objpts) < 8:
        sys.exit("need at least 8 good images")
    rms, K, dist, _, _ = cv2.calibrateCamera(objpts, imgpts, size, None, None)
    res = {"width": size[0], "height": size[1], "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
           "dist_coeffs": dist.ravel().tolist(), "rms_px": rms, "images": len(objpts),
           "date": time.strftime("%Y-%m-%d")}
    out = ROOT / "config" / "camera_calibration.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--n", type=int, default=25)
    s = sub.add_parser("solve")
    s.add_argument("--images", default=str(ROOT / "data" / "calib"))
    s.add_argument("--cols", type=int, default=9)
    s.add_argument("--rows", type=int, default=6)
    s.add_argument("--square", type=float, default=0.025)
    a = ap.parse_args()
    capture(a) if a.cmd == "capture" else solve(a)


if __name__ == "__main__":
    main()
