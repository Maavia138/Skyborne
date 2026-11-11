"""
Camera back-ends. Every back-end returns (frame_bgr, t_exposure) where
t_exposure is on the same monotonic clock as the telemetry buffer, so the
pose can be interpolated at the exact moment the photo was exposed.

* Picamera2Camera - Raspberry Pi Camera Module v2 on the Pi 5 (flight)
* OpenCVCamera    - any USB webcam (bench testing)
The simulated camera lives in simulation/sim_camera.py.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from config.settings import CameraConfig

log = logging.getLogger("camera")


class CameraBase:
    width: int
    height: int

    def start(self):  # pragma: no cover - interface
        return self

    def capture(self) -> tuple[np.ndarray, float]:  # pragma: no cover - interface
        raise NotImplementedError

    def stop(self):
        pass


class Picamera2Camera(CameraBase):
    """
    Full-resolution stills from the IMX219 with a fixed fast shutter.
    Picamera2's "RGB888" format is laid out B,G,R in memory -> directly usable by OpenCV.
    """

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.width, self.height = cfg.width, cfg.height
        self.cam = None

    def start(self):
        from picamera2 import Picamera2  # only available on the Pi
        self.cam = Picamera2()
        config = self.cam.create_still_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}, buffer_count=2)
        self.cam.configure(config)
        controls = {"ExposureTime": int(self.cfg.exposure_us), "AwbEnable": True}
        if self.cfg.analogue_gain:
            controls["AnalogueGain"] = float(self.cfg.analogue_gain)
        self.cam.set_controls(controls)
        self.cam.start()
        time.sleep(1.0)  # let AGC / AWB settle
        log.info("Picamera2 started %dx%d, shutter %d us", self.width, self.height, self.cfg.exposure_us)
        return self

    def capture(self):
        req = self.cam.capture_request()
        try:
            frame = req.make_array("main")
            md = req.get_metadata()
        finally:
            req.release()
        return frame, self._exposure_time(md)

    def _exposure_time(self, md: dict) -> float:
        """Convert the libcamera sensor timestamp (ns) to our monotonic clock."""
        now_mono = time.monotonic_ns()
        sensor_ns = md.get("SensorTimestamp")
        if sensor_ns:
            for clk in (getattr(time, "CLOCK_BOOTTIME", None), time.CLOCK_MONOTONIC):
                if clk is None:
                    continue
                age = time.clock_gettime_ns(clk) - sensor_ns
                if 0 <= age < 2_000_000_000:
                    exp_ns = md.get("ExposureTime", 0) * 1000 / 2   # mid-exposure
                    return (now_mono - age + exp_ns) / 1e9
        return now_mono / 1e9 - self.cfg.capture_latency_s

    def stop(self):
        if self.cam:
            self.cam.stop()
            self.cam.close()


class OpenCVCamera(CameraBase):
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.cap = None
        self.width, self.height = cfg.width, cfg.height

    def start(self):
        self.cap = cv2.VideoCapture(self.cfg.opencv_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("USB camera not available")
        self.height, self.width = frame.shape[:2]
        log.info("OpenCV camera %dx%d", self.width, self.height)
        return self

    def capture(self):
        ok, frame = self.cap.read()
        t = time.monotonic() - self.cfg.capture_latency_s
        if not ok:
            raise RuntimeError("camera read failed")
        return frame, t

    def stop(self):
        if self.cap:
            self.cap.release()


def create_camera(cfg: CameraConfig) -> CameraBase:
    if cfg.backend == "picamera2":
        return Picamera2Camera(cfg)
    if cfg.backend == "opencv":
        return OpenCVCamera(cfg)
    raise ValueError(f"camera backend '{cfg.backend}' must be created by the simulator")
