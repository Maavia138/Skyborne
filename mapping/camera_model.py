"""
Pinhole camera model + drone attitude -> ground geometry.

Frames
------
NED   : x North, y East, z Down (ArduPilot convention)
body  : x forward (nose), y right, z down
camera: x image-right, y image-down, z optical axis

For a camera pointing straight down with the image top towards the nose:
    image right -> body right   (x_c -> y_b)
    image down  -> body back    (y_c -> -x_b)
    optical axis-> body down    (z_c -> z_b)

A pixel (u, v) becomes a ray d_ned = R_nb * R_bc * K^-1 [u, v, 1]^T, which we
intersect with flat ground H metres below the camera:  P = (H / d_down) * d_ned.
This is the "exact" version of the D = (u - cx) * H / f formula: it also
handles roll, pitch, yaw, mount angles and lens distortion.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from config.settings import CameraConfig, nominal_fx

R_BODY_CAM = np.array([[0.0, -1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]])


def rot_ned_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body -> NED rotation, ZYX Euler (yaw, pitch, roll) in radians."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


class CameraModel:
    def __init__(self, width, height, fx, fy, cx, cy, dist=None, mount_rpy_deg=(0, 0, 0)):
        self.width, self.height = int(width), int(height)
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)
        self.has_dist = bool(np.any(np.abs(self.dist) > 1e-9))
        self.mount_rpy_deg = tuple(mount_rpy_deg)
        r, p, y = (math.radians(a) for a in mount_rpy_deg)
        self.R_bc = rot_ned_body(r, p, y) @ R_BODY_CAM
        self._undistort_maps = None

    # ------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, c: CameraConfig) -> "CameraModel":
        fx = c.fx or nominal_fx(c)
        fy = c.fy or fx
        cx = c.cx if c.cx is not None else c.width / 2.0
        cy = c.cy if c.cy is not None else c.height / 2.0
        return cls(c.width, c.height, fx, fy, cx, cy, c.dist_coeffs,
                   (c.mount_roll_deg, c.mount_pitch_deg, c.mount_yaw_deg))

    def scaled_to(self, width: int, height: int) -> "CameraModel":
        sx, sy = width / self.width, height / self.height
        return CameraModel(width, height, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                           self.dist, self.mount_rpy_deg)

    def pinhole(self) -> "CameraModel":
        """Same camera without distortion (use after undistorting the image)."""
        return CameraModel(self.width, self.height, self.fx, self.fy, self.cx, self.cy,
                           None, self.mount_rpy_deg)

    # ----------------------------------------------------------------- basics
    def gsd(self, height_m: float) -> float:
        return height_m / self.fx

    def footprint(self, height_m: float) -> tuple[float, float]:
        """(across-track width, along-track length) in metres for a nadir view."""
        return self.width * height_m / self.fx, self.height * height_m / self.fy

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        if not self.has_dist:
            return img
        if self._undistort_maps is None:
            self._undistort_maps = cv2.initUndistortRectifyMap(
                self.K, self.dist, None, self.K, (self.width, self.height), cv2.CV_16SC2)
        return cv2.remap(img, *self._undistort_maps, cv2.INTER_LINEAR)

    def normalized(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.has_dist:
            return cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        return np.column_stack(((pts[:, 0] - self.cx) / self.fx, (pts[:, 1] - self.cy) / self.fy))

    # --------------------------------------------------------------- geometry
    def rotation(self, roll, pitch, yaw) -> np.ndarray:
        """Camera -> NED rotation."""
        return rot_ned_body(roll, pitch, yaw) @ self.R_bc

    def ground_offsets(self, pts, roll, pitch, yaw, height_m):
        """
        Pixels -> ground offsets (north, east) in metres relative to the point
        directly below the camera. Returns (offsets (N,2), valid mask (N,)).
        """
        xy = self.normalized(pts)
        rays = np.column_stack((xy, np.ones(len(xy))))
        d = rays @ self.rotation(roll, pitch, yaw).T
        valid = d[:, 2] > 0.05                       # ray must point at the ground
        t = np.where(valid, height_m / np.where(valid, d[:, 2], 1.0), 0.0)
        return np.column_stack((t * d[:, 0], t * d[:, 1])), valid

    def project_offsets(self, ne_offsets, roll, pitch, yaw, height_m):
        """Ground offsets (north, east) relative to the camera nadir -> pixels."""
        ne = np.asarray(ne_offsets, dtype=np.float64).reshape(-1, 2)
        p_ned = np.column_stack((ne, np.full(len(ne), height_m)))
        p_cam = p_ned @ self.rotation(roll, pitch, yaw)    # R^T applied row-wise
        valid = p_cam[:, 2] > 0.1
        z = np.where(valid, p_cam[:, 2], 1.0)
        if self.has_dist:
            uv, _ = cv2.projectPoints(p_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
            return uv.reshape(-1, 2), valid
        return np.column_stack((self.fx * p_cam[:, 0] / z + self.cx,
                                self.fy * p_cam[:, 1] / z + self.cy)), valid

    def off_nadir_deg(self, pts, roll, pitch, yaw) -> np.ndarray:
        xy = self.normalized(pts)
        rays = np.column_stack((xy, np.ones(len(xy))))
        d = rays @ self.rotation(roll, pitch, yaw).T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return np.degrees(np.arccos(np.clip(d[:, 2], -1, 1)))
