"""
Simulated cameras.

SimCamera      - renders the synthetic world from a given true pose (sim mode).
SitlSimCamera  - "software-in-the-loop": takes the pose from ArduPilot SITL
                 over MAVLink and renders the synthetic world from it, so the
                 full onboard program (MAVLink thread, capture trigger, timing,
                 finalize on disarm, USB export) runs exactly as in flight.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from camera.camera_capture import CameraBase
from config.settings import Settings
from mapping.camera_model import CameraModel
from simulation.world import SyntheticWorld

log = logging.getLogger("simcam")


class SimCamera:
    def __init__(self, world: SyntheticWorld, cam: CameraModel, lens_h: float, seed: int = 0):
        self.world = world
        self.cam = cam
        self.lens_h = lens_h
        self.rng = np.random.default_rng(seed)

    def render(self, true_pose):
        return self.world.render(true_pose, self.cam, true_pose.alt + self.lens_h, self.rng)


class SitlSimCamera(CameraBase):
    def __init__(self, cfg: Settings, telemetry, survey_north_offset_m: float | None = None):
        self.cfg = cfg
        self.tel = telemetry
        full = CameraModel.from_config(cfg.camera)
        s = cfg.sim.render_scale
        self.cam = full.scaled_to(int(full.width * s), int(full.height * s))
        self.width, self.height = self.cam.width, self.cam.height
        self.offset = survey_north_offset_m if survey_north_offset_m is not None else cfg.sim.survey_size_m[0] / 2 + 20
        self.world: SyntheticWorld | None = None
        self.rng = np.random.default_rng(cfg.sim.seed)

    def start(self):
        log.info("Waiting for SITL position to anchor the synthetic world ...")
        while self.tel.latest() is None:
            time.sleep(0.2)
        p = self.tel.latest()
        from mapping.gps_converter import LocalFrame
        home = LocalFrame(p.lat, p.lon)
        center = home.to_latlon(self.offset, 0.0)
        self.world = SyntheticWorld(self.cfg.sim, center_latlon=center)
        t = self.world.tent
        log.info("Synthetic world centred %.0f m north of home; tent truth %.7f, %.7f", self.offset, t["lat"], t["lon"])
        return self

    def capture(self):
        t = time.monotonic()
        pose = self.tel.pose_at(t) or self.tel.latest()
        img = self.world.render(pose, self.cam, pose.alt + self.cfg.camera.lens_height_on_ground_m, self.rng)
        return img, t
