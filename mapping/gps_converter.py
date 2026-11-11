"""
GPS <-> local metric coordinates.

We work in a local North-East (NE) tangent plane centred on a reference point
(home / first GPS fix). Over a few kilometres the error of this approximation is
millimetres, far below GPS error. Radii come from the WGS-84 ellipsoid.
"""
from __future__ import annotations

import math

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def earth_radii(lat_deg: float) -> tuple[float, float]:
    """(meridional radius M, prime-vertical radius N) at latitude, metres."""
    s = math.sin(math.radians(lat_deg))
    d = 1.0 - WGS84_E2 * s * s
    return WGS84_A * (1 - WGS84_E2) / d ** 1.5, WGS84_A / math.sqrt(d)


class LocalFrame:
    """Flat-earth North/East frame around (lat0, lon0)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.rm, self.rn = earth_radii(self.lat0)
        self.k_lon = self.rn * math.cos(math.radians(self.lat0))

    def to_ne(self, lat, lon):
        n = np.radians(np.asarray(lat, dtype=np.float64) - self.lat0) * self.rm
        e = np.radians(np.asarray(lon, dtype=np.float64) - self.lon0) * self.k_lon
        return (float(n), float(e)) if np.ndim(n) == 0 else (n, e)

    def to_latlon(self, n, e):
        lat = self.lat0 + np.degrees(np.asarray(n, dtype=np.float64) / self.rm)
        lon = self.lon0 + np.degrees(np.asarray(e, dtype=np.float64) / self.k_lon)
        return (float(lat), float(lon)) if np.ndim(lat) == 0 else (lat, lon)

    def to_dict(self) -> dict:
        return {"lat0": self.lat0, "lon0": self.lon0}


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def to_dms(value: float, is_lat: bool) -> str:
    hemi = ("N" if value >= 0 else "S") if is_lat else ("E" if value >= 0 else "W")
    v = abs(value)
    d = int(v)
    m = int((v - d) * 60)
    s = (v - d - m / 60) * 3600
    return f"{d}°{m:02d}'{s:05.2f}\"{hemi}"


def to_dms_ascii(value: float, is_lat: bool) -> str:
    return to_dms(value, is_lat).replace("°", " deg ")


def point_in_polygon_ne(n: float, e: float, poly_ne: np.ndarray, margin_m: float = 0.0) -> bool:
    """True if (n,e) is inside the polygon or within margin_m of it."""
    import cv2
    contour = np.asarray(poly_ne, dtype=np.float32)[:, ::-1].reshape(-1, 1, 2)  # (e, n)
    d = cv2.pointPolygonTest(contour, (float(e), float(n)), True)
    return d >= -margin_m
