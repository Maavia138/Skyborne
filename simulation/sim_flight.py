"""
Simulated survey flight with realistic sensor errors.

Flight: take-off south of the survey area -> lawn-mower pattern (lines
north/south, spacing from the camera footprint and side overlap) -> return.
Truth vs measurement:
    true_pose()     - where the drone really is (used to render the camera image)
    measured_pose() - what the flight controller reports (GPS bias + jitter,
                      altitude bias, attitude noise, compass bias). The pipeline
                      only ever sees this, so the simulator measures the real
                      end-to-end geolocation error.
"""
from __future__ import annotations

import math

import numpy as np

from communication.telemetry import Pose, VehicleStatus
from config.settings import Settings
from mapping.camera_model import CameraModel
from simulation.world import SyntheticWorld


class SimFlight:
    def __init__(self, cfg: Settings, world: SyntheticWorld, cam: CameraModel, seed: int | None = None):
        s = cfg.sim
        self.cfg = cfg
        self.world = world
        self.alt = s.altitude_m
        self.v = s.speed_mps
        self.rng = np.random.default_rng((s.seed if seed is None else seed) + 1000)
        fw, _ = cam.footprint(self.alt)
        self.spacing = fw * (1 - cfg.capture.side_overlap)
        sn, se = s.survey_size_m
        n_lines = int(math.ceil(se / self.spacing)) + 1
        es = np.linspace(-se / 2, se / 2, n_lines)
        over = 8.0
        home = (-sn / 2 - 20.0, 0.0)
        wps = [(home[0], home[1], 0.0), (home[0], home[1], self.alt)]
        for i, e in enumerate(es):
            a, b = (-sn / 2 - over, sn / 2 + over) if i % 2 == 0 else (sn / 2 + over, -sn / 2 - over)
            wps += [(a, e, self.alt), (b, e, self.alt)]
        wps += [(home[0], home[1], self.alt), (home[0], home[1], 0.0)]
        self.waypoints = np.array(wps)
        self.seg_t = [0.0]
        self.headings = []
        hd = 0.0
        for p, q in zip(self.waypoints[:-1], self.waypoints[1:]):
            dh = math.hypot(q[0] - p[0], q[1] - p[1])
            dv = abs(q[2] - p[2])
            dur = max(dh / self.v, dv / 2.5, 0.1)
            if dh > 0.5:
                hd = math.atan2(q[1] - p[1], q[0] - p[0])
            self.headings.append(hd)
            self.seg_t.append(self.seg_t[-1] + dur)
        self.duration = self.seg_t[-1]
        self.lines = n_lines
        ph = self.rng.uniform(0, 2 * math.pi, 8)
        self._ph = ph

    def _state(self, t):
        t = float(np.clip(t, 0, self.duration))
        i = int(np.searchsorted(self.seg_t, t, side="right") - 1)
        i = min(i, len(self.waypoints) - 2)
        w = (t - self.seg_t[i]) / (self.seg_t[i + 1] - self.seg_t[i])
        p = self.waypoints[i] + w * (self.waypoints[i + 1] - self.waypoints[i])
        moving = math.hypot(*(self.waypoints[i + 1][:2] - self.waypoints[i][:2])) > 0.5
        return p, self.headings[i], (self.v if moving else 0.0)

    def true_pose(self, t) -> Pose:
        s = self.cfg.sim
        p, hd, gs = self._state(t)
        a = math.radians(s.wind_tilt_deg)
        ph = self._ph
        roll = a * (0.7 * math.sin(2 * math.pi * t / 6.1 + ph[0]) + 0.3 * math.sin(2 * math.pi * t / 2.3 + ph[1]))
        pitch = a * (0.7 * math.sin(2 * math.pi * t / 7.7 + ph[2]) + 0.3 * math.sin(2 * math.pi * t / 1.9 + ph[3])) \
            - math.radians(4.0) * (gs / 5.0)                    # nose down when flying forward
        yaw = hd + math.radians(3.0) * math.sin(2 * math.pi * t / 23.0 + ph[4])
        lat, lon = self.world.frame.to_latlon(p[0], p[1])
        return Pose(t, lat, lon, float(p[2]), roll, pitch, yaw, gs, 3, 14, 0.8)

    def measured_pose(self, t) -> Pose:
        s = self.cfg.sim
        tp = self.true_pose(t)
        n, e = self.world.frame.to_ne(tp.lat, tp.lon)
        ph = self._ph
        bn = s.gps_bias_m * (0.8 * math.sin(2 * math.pi * t / 240 + ph[5]) + 0.2 * math.sin(2 * math.pi * t / 37 + ph[6]))
        be = s.gps_bias_m * (0.8 * math.cos(2 * math.pi * t / 300 + ph[7]) + 0.2 * math.sin(2 * math.pi * t / 41 + ph[0]))
        n += bn + self.rng.normal(0, s.gps_noise_m)
        e += be + self.rng.normal(0, s.gps_noise_m)
        lat, lon = self.world.frame.to_latlon(n, e)
        an = math.radians(s.att_noise_deg)
        return Pose(t, lat, lon, tp.alt + s.alt_bias_m + self.rng.normal(0, 0.15),
                    tp.roll + self.rng.normal(0, an), tp.pitch + self.rng.normal(0, an),
                    tp.yaw + math.radians(s.yaw_bias_deg) + self.rng.normal(0, an),
                    tp.groundspeed, 3, 14, 0.8)

    def status(self, t) -> VehicleStatus:
        armed = 0.0 < t < self.duration - 0.5
        pct = 100 - 55 * t / max(self.duration, 1)
        return VehicleStatus(connected=True, armed=armed, mode="AUTO" if armed else "LAND",
                             battery_v=12.6 - 1.4 * t / max(self.duration, 1), battery_pct=pct, last_heartbeat=t)
