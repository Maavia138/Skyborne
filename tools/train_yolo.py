"""
Fine-tune YOLO11n (or YOLOv8n) to detect white tents, then export for the Pi.

Install on the laptop (NOT needed on the Pi):
    pip install ultralytics onnx onnxslim

    python tools/train_yolo.py --data datasets/tents/data.yaml --epochs 60
    -> models/tent_yolo.onnx   (copy this file to the Pi; onboard code uses onnxruntime/OpenCV)

Why start from COCO-pretrained weights: the backbone already knows edges,
textures and shapes, so a few hundred tent images are enough. Training from
scratch would need tens of thousands and days of GPU time.

Tips
  * Use a GPU if available (Google Colab free T4 works: ~1-2 min/epoch for 1500 images).
  * imgsz 640 matches the onboard inference size.
  * Mix synthetic + real images; validate ONLY on real photos before competition.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/tents/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt | yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
                project=str(ROOT / "runs"), name="tent", exist_ok=True, patience=20,
                degrees=180, flipud=0.5, fliplr=0.5, mosaic=1.0, scale=0.4, hsv_v=0.3, close_mosaic=10)
    best = ROOT / "runs" / "tent" / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=a.data, imgsz=a.imgsz)
    print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)
    onnx = YOLO(str(best)).export(format="onnx", imgsz=a.imgsz, opset=12, simplify=True, dynamic=False)
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(onnx, ROOT / "models" / "tent_yolo.onnx")
    print("Exported ->", ROOT / "models" / "tent_yolo.onnx")
    # Optional, often fastest on Raspberry Pi: NCNN format
    # YOLO(str(best)).export(format="ncnn", imgsz=a.imgsz)


if __name__ == "__main__":
    main()
