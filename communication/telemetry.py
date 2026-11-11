"""
Telemetry data types and a thread-safe buffer that interpolates the drone pose
at the exact moment a photo was exposed.

Position and attitude arrive as separate MAVLink messages at ~10 Hz. A photo
taken between two messages gets a linearly interpolated pose (yaw interpolated
along the shortest arc). At 5 m/s this removes up to 0.5 m of tagging error.
"""
from __future__ import annotations

import bisect
import math
import threading
from collections import deque
from dataclasses import asdict, dataclass

CSV_FIELDS = ["t", "lat", "lon", "alt", "roll_deg", "pitch_deg", "yaw_deg",
              "groundspeed", "gps_fix", "satellites", "hdop", "rangefinder_m"]


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Pose:
    t: float                 # monotonic seconds
    lat: float
    lon: float
    alt: float               # metres above home (relative altitude)
    roll: float              # radians
    pitch: float
    yaw: float               # radians, 0 = north, positive clockwise
    groundspeed: float = 0.0
    gps_fix: int = 3
    satellites: int = 10
    hdop: float = 1.0
    rangefinder_m: float = float("nan")

    def tilt_deg(self) -> float:
        c = math.cos(self.roll) * math.cos(self.pitch)
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    def to_row(self) -> dict:
        return {"t": round(self.t, 3), "lat": f"{self.lat:.8f}", "lon": f"{self.lon:.8f}",
                "alt": round(self.alt, 2), "roll_deg": round(math.degrees(self.roll), 2),
                "pitch_deg": round(math.degrees(self.pitch), 2),
                "yaw_deg": round(math.degrees(self.yaw) % 360, 2),
                "groundspeed": round(self.groundspeed, 2), "gps_fix": self.gps_fix,
                "satellites": self.satellites, "hdop": round(self.hdop, 2),
                "rangefinder_m": "" if math.isnan(self.rangefinder_m) else round(self.rangefinder_m, 2)}

    @staticmethod
    def from_row(r: dict) -> "Pose":
        rf = r.get("rangefinder_m", "")
        return Pose(t=float(r["t"]), lat=float(r["lat"]), lon=float(r["lon"]), alt=float(r["alt"]),
                    roll=math.radians(float(r["roll_deg"])), pitch=math.radians(float(r["pitch_deg"])),
                    yaw=math.radians(float(r["yaw_deg"])), groundspeed=float(r.get("groundspeed", 0) or 0),
                    gps_fix=int(float(r.get("gps_fix", 3) or 3)), satellites=int(float(r.get("satellites", 0) or 0)),
                    hdop=float(r.get("hdop", 1) or 1), rangefinder_m=float(rf) if rf not in ("", None) else float("nan"))


@dataclass
class VehicleStatus:
    connected: bool = False
    armed: bool = False
    mode: str = "UNKNOWN"
    battery_v: float = float("nan")
    battery_pct: float = -1.0
    last_heartbeat: float = 0.0

    def copy(self) -> "VehicleStatus":
        return VehicleStatus(**asdict(self))


class TelemetryBuffer:
    def __init__(self, maxlen: int = 4000, max_gap_s: float = 0.6):
        self._lock = threading.Lock()
        self._pos: deque = deque(maxlen=maxlen)   # (t, lat, lon, alt, gs)
        self._att: deque = deque(maxlen=maxlen)   # (t, roll, pitch, yaw)
        self._gps = (0, 0, 99.0)
        self._range = (0.0, float("nan"))
        self.max_gap_s = max_gap_s

    def add_position(self, t, lat, lon, alt, groundspeed=0.0):
        with self._lock:
            if self._pos and t <= self._pos[-1][0]:
                return
            self._pos.append((t, lat, lon, alt, groundspeed))

    def add_attitude(self, t, roll, pitch, yaw):
        with self._lock:
            if self._att and t <= self._att[-1][0]:
                return
            self._att.append((t, roll, pitch, yaw))

    def set_gps_quality(self, fix, sats, hdop):
        with self._lock:
            self._gps = (fix, sats, hdop)

    def set_rangefinder(self, t, dist_m):
        with self._lock:
            self._range = (t, dist_m)

    @staticmethod
    def _interp(samples, t, max_gap, angle_idx=()):
        if not samples:
            return None
        if t >= samples[-1][0]:
            return samples[-1] if t - samples[-1][0] <= max_gap else None
        if t <= samples[0][0]:
            return samples[0] if samples[0][0] - t <= max_gap else None
        times = [s[0] for s in samples]
        i = bisect.bisect_left(times, t)
        a, b = samples[i - 1], samples[i]
        if b[0] - a[0] > 2 * max_gap:
            return None
        w = (t - a[0]) / (b[0] - a[0])
        out = [t]
        for k in range(1, len(a)):
            if k in angle_idx:
                out.append(wrap_pi(a[k] + w * wrap_pi(b[k] - a[k])))
            else:
                out.append(a[k] + w * (b[k] - a[k]))
        return tuple(out)

    def pose_at(self, t: float) -> Pose | None:
        with self._lock:
            p = self._interp(self._pos, t, self.max_gap_s)
            a = self._interp(self._att, t, self.max_gap_s, angle_idx=(3,))
            fix, sats, hdop = self._gps
            rt, rd = self._range
        if p is None or a is None:
            return None
        rf = rd if abs(t - rt) < 0.5 else float("nan")
        return Pose(t, p[1], p[2], p[3], a[1], a[2], a[3], p[4], fix, sats, hdop, rf)

    def latest(self) -> Pose | None:
        with self._lock:
            if not self._pos or not self._att:
                return None
            t = min(self._pos[-1][0], self._att[-1][0])
        return self.pose_at(t)
