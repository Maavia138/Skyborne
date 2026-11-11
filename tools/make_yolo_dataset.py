"""
Generate a YOLO-format training dataset of aerial tent images from the simulator.

Each image is a random camera view (altitude 20-45 m, random roll/pitch/yaw,
random lighting) over a freshly randomised world. ~70 % contain the white tent
(ridge or dome style, random size/orientation); all contain hard negatives
(white car, white roof, concrete, road paint, stones, orange tent).

    python tools/make_yolo_dataset.py --out datasets/tents --n 1500

IMPORTANT: synthetic images teach the model the geometry and the negatives,
but you MUST add real photos (your own test flights over a white tent, plus
public aerial datasets) before trusting it. Label real photos with any YOLO
labelling tool (e.g. Label Studio, CVAT, Roboflow) into the same folders.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from communication.telemetry import Pose  # noqa: E402
from config.settings import Settings  # noqa: E402
from mapping.camera_model import CameraModel  # noqa: E402
from simulation.world import SyntheticWorld  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/tents")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--views-per-world", type=int, default=12)
    ap.add_argument("--size", type=int, default=960, help="saved image width")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()

    cfg = Settings.load()
    cfg.sim.world_size_m = [120.0, 120.0]
    cfg.sim.survey_size_m = [90.0, 90.0]
    full = CameraModel.from_config(cfg.camera)
    cam = full.scaled_to(a.size, int(a.size * full.height / full.width))
    out = Path(a.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    i = 0
    while i < a.n:
        cfg.sim.tent_size_m = [float(rng.uniform(1.8, 3.5)), float(rng.uniform(1.4, 2.6))]
        cfg.sim.tent_ne = None
        has_tent = rng.random() < 0.85
        world = SyntheticWorld(cfg.sim, seed=int(rng.integers(1e9)), place_tent=has_tent,
                               tent_style=rng.choice(["ridge", "dome"]))
        for _ in range(a.views_per_world):
            if i >= a.n:
                break
            alt = rng.uniform(20, 45)
            target = world.tent if (has_tent and rng.random() < 0.8) else None
            if target is not None:
                gn, ge = target["n"] + rng.uniform(-12, 12), target["e"] + rng.uniform(-12, 12)
            else:
                gn, ge = rng.uniform(-40, 40), rng.uniform(-40, 40)
            lat, lon = world.frame.to_latlon(gn, ge)
            pose = Pose(0, lat, lon, alt, math.radians(rng.normal(0, 4)), math.radians(rng.normal(0, 4)),
                        rng.uniform(-math.pi, math.pi))
            img = world.render(pose, cam, alt, rng)
            # random photometric changes: exposure, white balance, blur, JPEG
            img = np.clip(img.astype(np.float32) * rng.uniform(0.75, 1.2) * rng.uniform(0.93, 1.07, 3), 0, 255).astype(np.uint8)
            if rng.random() < 0.3:
                img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
            labels = []
            for obj in world.objects:
                if obj["type"] != "white_tent":
                    continue
                bb = world.project_object(obj, pose, cam, alt)
                if bb is None:
                    continue
                x1, y1, x2, y2 = bb
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(cam.width, x2), min(cam.height, y2)
                if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                    continue
                vis = (cx2 - cx1) * (cy2 - cy1) / ((x2 - x1) * (y2 - y1))
                if vis < 0.4:
                    continue
                labels.append(f"0 {(cx1 + cx2) / 2 / cam.width:.6f} {(cy1 + cy2) / 2 / cam.height:.6f} "
                              f"{(cx2 - cx1) / cam.width:.6f} {(cy2 - cy1) / cam.height:.6f}")
            split = "val" if rng.random() < a.val_frac else "train"
            name = f"syn_{i:05d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(75, 96))])
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(labels))
            i += 1
        print(f"{i}/{a.n}", end="\r")
    (out / "data.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
                                   f"names:\n  0: white_tent\n")
    print(f"\nDataset written to {out} ({a.n} images)")


if __name__ == "__main__":
    main()
