"""
YOLO tent detector running an exported ONNX model on the CPU.

Why YOLO11n/YOLOv8n and not a transformer (RT-DETR, DETR, ViT)?
  * ~2.6 M parameters vs 20-40 M; ~250-450 ms per 640 px frame on a Pi 5 CPU,
    which fits our 1 frame/s budget. RT-DETR-R18 takes several seconds per frame.
  * Small objects (a tent is 40-60 px at 640 px input) are well handled by
    YOLO's high-resolution P3 head.
Train with tools/train_yolo.py, export to ONNX, copy to models/tent_yolo.onnx.

Back-ends tried in order: onnxruntime (fastest on ARM) -> OpenCV DNN.
Supports the standard Ultralytics output (1, 4+nc, N) and NMS-free
end-to-end output (1, N, 6).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config.settings import DetectionConfig
from detection.tent_detector import Detection


class YoloDetector:
    def __init__(self, model_path: str | Path, cfg: DetectionConfig):
        self.cfg = cfg
        self.imgsz = cfg.yolo_imgsz
        self.names = cfg.yolo_class_names
        path = str(model_path)
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
            self.backend = "onnxruntime"
        except ImportError:
            self.net = cv2.dnn.readNetFromONNX(path)
            self.backend = "opencv-dnn"

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        px, py = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        return canvas, r, px, py

    def _infer(self, img):
        lb, r, px, py = self._letterbox(img)
        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, swapRB=True)
        if self.backend == "onnxruntime":
            out = self.sess.run(None, {self.inp: blob})[0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()
        out = np.squeeze(out, 0)
        if out.ndim == 2 and out.shape[1] == 6 and out.shape[0] > out.shape[1]:   # end-to-end
            boxes, conf, cls = out[:, :4], out[:, 4], out[:, 5].astype(int)
            keep = conf >= self.cfg.yolo_conf
            boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        else:
            out = out.T                                                         # (N, 4+nc)
            scores = out[:, 4:]
            cls = scores.argmax(1)
            conf = scores[np.arange(len(scores)), cls]
            keep = conf >= self.cfg.yolo_conf
            b, conf, cls = out[keep, :4], conf[keep], cls[keep]
            boxes = np.column_stack((b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2))
            if len(boxes):
                idx = cv2.dnn.NMSBoxes(np.column_stack((boxes[:, :2], boxes[:, 2:] - boxes[:, :2])).tolist(),
                                       conf.tolist(), self.cfg.yolo_conf, self.cfg.yolo_iou)
                idx = np.array(idx).reshape(-1)
                boxes, conf, cls = boxes[idx], conf[idx], cls[idx]
        boxes = (boxes - [px, py, px, py]) / r
        return boxes, conf, cls

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        H, W = img_bgr.shape[:2]
        n = max(1, self.cfg.yolo_tiles)
        tiles = []
        if n == 1:
            tiles.append((0, 0, img_bgr))
        else:
            tw, th = int(W / n * 1.15), int(H / n * 1.15)       # 15 % overlap
            for i in range(n):
                for j in range(n):
                    x0 = min(W - tw, int(i * W / n)); y0 = min(H - th, int(j * H / n))
                    tiles.append((x0, y0, img_bgr[y0:y0 + th, x0:x0 + tw]))
        dets: list[Detection] = []
        for x0, y0, tile in tiles:
            boxes, conf, cls = self._infer(tile)
            for b, c, k in zip(boxes, conf, cls):
                if int(k) != self.cfg.yolo_tent_class:
                    continue
                x1, y1, x2, y2 = np.clip(b + [x0, y0, x0, y0], 0, [W, H, W, H])
                trunc = bool(x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2)
                dets.append(Detection(bbox=[float(x1), float(y1), float(x2), float(y2)],
                                      confidence=float(c) * (0.7 if trunc else 1.0), source="yolo",
                                      cls=self.names[int(k)] if int(k) < len(self.names) else str(k),
                                      truncated=trunc))
        if n > 1 and dets:   # merge duplicates from overlapping tiles
            keep = cv2.dnn.NMSBoxes([[d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]] for d in dets],
                                    [d.confidence for d in dets], 0.0, 0.5)
            dets = [dets[i] for i in np.array(keep).reshape(-1)]
        return dets
