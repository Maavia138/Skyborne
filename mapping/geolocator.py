"""
Tent geolocation: pixel -> latitude/longitude, then fusion of many sightings.

Single photo
    tent pixel -> camera ray (with roll, pitch, yaw, lens model) -> intersect
    flat ground -> north/east offset from the drone -> lat/lon.

Many photos
    With 75 % forward and 40 % side overlap the tent appears in 4-10 photos.
    Sightings within `cluster_radius_m` are grouped; the cluster with the best
    combined evidence is the tent. Its position is the inverse-variance
    weighted mean after outlier rejection. Random errors (attitude noise,
    GPS jitter) shrink with more sightings; GPS bias does not, so it is added
    back when we report the 95 % error radius.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from config.settings import GeolocationConfig
from mapping.camera_model import CameraModel
from mapping.gps_converter import LocalFrame


@dataclass
class Sighting:
    frame_id: int
    image: str
    t: float
    confidence: float
    source: str
    n: float                       # GPS-frame north/east (metres, local frame)
    e: float
    map_n: float                   # map-aligned position (after mosaic correction)
    map_e: float
    sigma_m: float                 # 1-sigma random error of this sighting
    off_nadir_deg: float
    corners_ne: list               # tent bbox corners on the ground (map-aligned)
    bbox: list
    altitude: float
    truncated: bool = False
    size_m: list = field(default_factory=list)


@dataclass
class TargetEstimate:
    lat: float
    lon: float
    n: float
    e: float
    map_n: float
    map_e: float
    confidence: float
    sightings: int
    error_95_m: float
    sigma_random_m: float
    spread_m: float
    best: Sighting
    members: list

    def to_dict(self) -> dict:
        d = asdict(self)
        d["best"] = asdict(self.best)
        d["members"] = [asdict(m) for m in self.members]
        return d


def locate_pixels(cam: CameraModel, frame: LocalFrame, pose, height_m, pts):
    """Pixels (N,2) -> absolute (north, east) in the local frame; also valid mask."""
    off, valid = cam.ground_offsets(pts, pose.roll, pose.pitch, pose.yaw, height_m)
    n0, e0 = frame.to_ne(pose.lat, pose.lon)
    return np.column_stack((n0 + off[:, 0], e0 + off[:, 1])), valid


def sighting_sigma(cfg: GeolocationConfig, height_m: float, horiz_dist_m: float, off_nadir_deg: float) -> float:
    att = height_m * math.radians(cfg.att_sigma_deg) / max(0.2, math.cos(math.radians(off_nadir_deg)) ** 2)
    yaw = horiz_dist_m * math.radians(cfg.yaw_sigma_deg)
    alt = cfg.alt_sigma_m * math.tan(math.radians(off_nadir_deg))
    return math.sqrt(att ** 2 + yaw ** 2 + alt ** 2 + cfg.gps_noise_m ** 2)


class TargetTracker:
    def __init__(self, cfg: GeolocationConfig, frame: LocalFrame | None = None):
        self.cfg = cfg
        self.frame = frame
        self.sightings: list[Sighting] = []

    def add(self, s: Sighting):
        self.sightings.append(s)

    def _clusters(self) -> list[list[Sighting]]:
        clusters: list[list[Sighting]] = []
        centers: list[np.ndarray] = []
        for s in sorted(self.sightings, key=lambda x: -x.confidence):
            p = np.array([s.n, s.e])
            if centers:
                d = [np.linalg.norm(p - c) for c in centers]
                k = int(np.argmin(d))
                if d[k] <= self.cfg.cluster_radius_m:
                    clusters[k].append(s)
                    w = np.array([m.confidence for m in clusters[k]])
                    pts = np.array([[m.n, m.e] for m in clusters[k]])
                    centers[k] = (pts * w[:, None]).sum(0) / w.sum()
                    continue
            clusters.append([s])
            centers.append(p)
        return clusters

    @staticmethod
    def cluster_score(members: list[Sighting]) -> float:
        conf = sorted((m.confidence * (0.4 if m.truncated else 1.0) for m in members), reverse=True)
        top = float(np.mean(conf[:5]))
        n_eff = sum(1 for m in members if not m.truncated) + 0.3 * sum(1 for m in members if m.truncated)
        return top * min(1.0, math.sqrt(n_eff / 3.0))

    def candidates(self) -> list[TargetEstimate]:
        out = [self._estimate(c) for c in self._clusters()]
        out = [e for e in out if e is not None]
        return sorted(out, key=lambda e: -e.confidence)

    def best(self) -> TargetEstimate | None:
        c = self.candidates()
        return c[0] if c else None

    def _estimate(self, members: list[Sighting]) -> TargetEstimate | None:
        if not members:
            return None
        full = [m for m in members if not m.truncated] or members
        pts = np.array([[m.n, m.e] for m in full])
        mpts = np.array([[m.map_n, m.map_e] for m in full])
        w = np.array([m.confidence / max(m.sigma_m, 0.05) ** 2 for m in full])
        # robust: drop sightings > 3 MAD from the median
        med = np.median(pts, 0)
        d = np.linalg.norm(pts - med, axis=1)
        mad = max(np.median(d), 0.3)
        keep = d <= 3 * mad + 0.5
        pts, mpts, w = pts[keep], mpts[keep], w[keep]
        kept = [m for m, k in zip(full, keep) if k]
        pos = (pts * w[:, None]).sum(0) / w.sum()
        mpos = (mpts * w[:, None]).sum(0) / w.sum()
        sig = np.array([m.sigma_m for m in kept])
        sigma_rand = math.sqrt(1.0 / np.sum(1.0 / sig ** 2))
        spread = float(np.sqrt(np.mean(np.sum((pts - pos) ** 2, 1)))) if len(pts) > 1 else 0.0
        # 2-D 95 % radius  = 2.45 sigma (Rayleigh); GPS bias does not average out
        sigma_tot = math.sqrt(self.cfg.gps_sigma_m ** 2 + sigma_rand ** 2 + (spread / math.sqrt(len(pts))) ** 2)
        best = min(kept, key=lambda m: (m.truncated, -m.confidence + m.off_nadir_deg / 100.0))
        lat, lon = self.frame.to_latlon(pos[0], pos[1])
        return TargetEstimate(lat=lat, lon=lon, n=float(pos[0]), e=float(pos[1]),
                              map_n=float(mpos[0]), map_e=float(mpos[1]),
                              confidence=round(self.cluster_score(members), 3), sightings=len(members),
                              error_95_m=round(2.45 * sigma_tot, 2), sigma_random_m=round(sigma_rand, 2),
                              spread_m=round(spread, 2), best=best, members=members)
