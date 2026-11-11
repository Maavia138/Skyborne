"""Unit tests:  python -m pytest -q"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from communication.telemetry import Pose, TelemetryBuffer, VehicleStatus  # noqa: E402
from config.settings import Settings  # noqa: E402
from detection.tent_detector import WhiteBlobDetector  # noqa: E402
from mapping.camera_model import CameraModel  # noqa: E402
from mapping.gps_converter import LocalFrame, haversine_m, point_in_polygon_ne  # noqa: E402
from mapping.image_mapper import GeoMosaic  # noqa: E402

LAT0, LON0 = 33.6844, 73.0479


@pytest.fixture(scope="module")
def cfg():
    return Settings()


@pytest.fixture(scope="module")
def cam(cfg):
    return CameraModel.from_config(cfg.camera).scaled_to(1640, 1232)


# ------------------------------------------------------------------ geodesy
def test_local_frame_roundtrip():
    f = LocalFrame(LAT0, LON0)
    lat, lon = f.to_latlon(123.4, -56.7)
    n, e = f.to_ne(lat, lon)
    assert abs(n - 123.4) < 1e-6 and abs(e + 56.7) < 1e-6
    # haversine uses a sphere; our frame uses the WGS-84 ellipsoid -> agree to ~0.3 %
    assert abs(haversine_m(LAT0, LON0, lat, lon) / math.hypot(123.4, 56.7) - 1) < 0.003


def test_point_in_polygon():
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert point_in_polygon_ne(5, 5, poly)
    assert not point_in_polygon_ne(15, 5, poly)
    assert point_in_polygon_ne(12, 5, poly, margin_m=3)


# ------------------------------------------------------------------ camera
def test_pi_camera_v2_gsd(cfg):
    full = CameraModel.from_config(cfg.camera)
    assert abs(full.gsd(30) * 100 - 1.105) < 0.01           # ~1.1 cm/px at 30 m (brief)
    w, h = full.footprint(30)
    assert 35 < w < 38 and 26 < h < 28


def test_nadir_centre_is_below_drone(cam):
    off, ok = cam.ground_offsets([[cam.cx, cam.cy]], 0, 0, 0.7, 30)
    assert ok.all() and np.allclose(off, 0, atol=1e-9)


def test_image_orientation(cam):
    # yaw 0: image top = north, image right = east
    off, _ = cam.ground_offsets([[cam.cx, 0], [cam.width, cam.cy]], 0, 0, 0, 30)
    assert off[0, 0] > 10 and abs(off[0, 1]) < 1e-6
    assert off[1, 1] > 10 and abs(off[1, 0]) < 1e-6
    # yaw 90 deg (facing east): image top = east
    off, _ = cam.ground_offsets([[cam.cx, 0]], 0, 0, math.pi / 2, 30)
    assert off[0, 1] > 10 and abs(off[0, 0]) < 1e-6


def test_simple_formula_matches(cam):
    u = cam.cx + 400
    off, _ = cam.ground_offsets([[u, cam.cy]], 0, 0, 0, 30)
    assert abs(off[0, 1] - 400 * 30 / cam.fx) < 1e-9          # D = (u - cx) H / f


def test_project_unproject_with_attitude(cam):
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], [cam.width, cam.height], (50, 2))
    r, p, y, h = 0.08, -0.05, 2.1, 31.0
    off, ok = cam.ground_offsets(pts, r, p, y, h)
    back, ok2 = cam.project_offsets(off, r, p, y, h)
    assert ok.all() and ok2.all() and np.abs(back - pts).max() < 1e-6


def test_pitch_shifts_footprint_forward(cam):
    # nose down (negative pitch) -> camera looks backwards? No: body z tilts forward when nose is UP.
    off, _ = cam.ground_offsets([[cam.cx, cam.cy]], 0, math.radians(10), 0, 30)
    assert off[0, 0] > 0                                         # nose up -> centre ray hits ground ahead


# --------------------------------------------------------------- telemetry
def test_buffer_interpolation():
    b = TelemetryBuffer()
    b.add_position(0.0, LAT0, LON0, 30, 5)
    b.add_position(1.0, LAT0 + 1e-4, LON0, 32, 5)
    b.add_attitude(0.0, 0, 0, math.radians(350))
    b.add_attitude(1.0, 0, 0, math.radians(10))
    p = b.pose_at(0.5)
    assert abs(p.alt - 31) < 1e-9 and abs(p.lat - (LAT0 + 5e-5)) < 1e-12
    assert abs(math.degrees(p.yaw) % 360) < 1e-6                 # shortest arc through north
    assert b.pose_at(5.0) is None                                # stale data rejected


# ---------------------------------------------------------------- detector
def _scene(gsd, objects):
    rng = np.random.default_rng(1)
    img = np.zeros((1232, 1640, 3), np.uint8)
    img[:] = (50, 120, 75)
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    for kind, cx, cy, w_m, h_m in objects:
        w, h = int(w_m / gsd), int(h_m / gsd)
        x1, y1 = cx - w // 2, cy - h // 2
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (240, 240, 240), -1)
        if kind == "car":
            cv2.rectangle(img, (x1 + w // 5, y1 + 3), (x1 + w // 3, y1 + h - 3), (40, 40, 40), -1)
            cv2.rectangle(img, (x1 + 3 * w // 4, y1 + 3), (x1 + 4 * w // 5, y1 + h - 3), (40, 40, 40), -1)
        if kind == "tent":
            cv2.rectangle(img, (x1, y1 + h // 2), (x1 + w, y1 + h), (212, 214, 214), -1)
    return img


def test_detector_finds_tent_rejects_decoys(cfg):
    gsd = 30 / CameraModel.from_config(cfg.camera).scaled_to(1640, 1232).fx
    img = _scene(gsd, [("tent", 500, 600, 2.4, 2.1), ("car", 1200, 300, 4.5, 1.8),
                       ("roof", 1200, 900, 12, 8), ("line", 300, 200, 3.0, 0.15)])
    dets = WhiteBlobDetector(cfg.detection).detect(img, gsd)
    assert dets, "tent not found"
    cx, cy = dets[0].center
    assert abs(cx - 500) < 15 and abs(cy - 600) < 15 and dets[0].confidence > 0.8
    assert all(d.confidence < 0.5 for d in dets[1:])
    assert 2.0 < dets[0].size_m[0] < 3.0


# ------------------------------------------------------------------ mosaic
def test_mosaic_alignment_reduces_gps_error(cfg):
    from simulation.world import SyntheticWorld
    sim = cfg.sim
    sim.world_size_m = [100.0, 100.0]
    sim.survey_size_m = [60.0, 60.0]
    world = SyntheticWorld(sim, seed=3)
    full = CameraModel.from_config(cfg.camera)
    cam = full.scaled_to(820, 616)
    f = world.frame
    m = GeoMosaic(f, 0.05, refine=True, gps_weight=0.0)
    true_n = [0.0, 7.0]
    gps_err = [(0.0, 0.0), (1.5, -1.0)]                     # second photo has 1.8 m GPS error
    for i in range(2):
        lat, lon = f.to_latlon(true_n[i], 0)
        img = world.render(Pose(0, lat, lon, 30, 0, 0, 0), cam, 30, effects=False)
        lat_m, lon_m = f.to_latlon(true_n[i] + gps_err[i][0], gps_err[i][1])
        info = m.add(img, cam, Pose(0, lat_m, lon_m, 30, 0, 0, 0), 30, i)
    dn, de = info["shift_m"]
    assert abs(dn + 1.5) < 0.3 and abs(de - 1.0) < 0.3, info


# ------------------------------------------------------------ end-to-end
def test_export_and_resume(tmp_path, cfg):
    from output.export_results import copy_to_usb, write_manifest
    src = tmp_path / "SKYBORNE_OUTPUT_x"
    src.mkdir()
    (src / "a.txt").write_text("hello")
    write_manifest(src)
    usb = tmp_path / "usb"
    usb.mkdir()
    ok, dst, msg = copy_to_usb(src, usb)
    assert ok and (dst / "a.txt").read_text() == "hello"


def test_capture_gate(cfg, tmp_path):
    from pipeline import SkyborneMission
    c = Settings()
    c.output.data_dir = str(tmp_path)
    c.output.results_dir = str(tmp_path / "res")
    c.detection.use_yolo = "off"
    m = SkyborneMission(c, tmp_path / "s")
    st = VehicleStatus(connected=True, armed=True)
    p = Pose(0, LAT0, LON0, 30, 0, 0, 0)
    assert m.should_capture(p, st, 0.0)[0]
    assert m.should_capture(p, st, 0.5)[1] == "interval"
    assert m.should_capture(p, st, 1.0)[1] == "spacing"
    lat, lon = LocalFrame(LAT0, LON0).to_latlon(8, 0)
    assert m.should_capture(Pose(0, lat, lon, 30, 0, 0, 0), st, 1.5)[0]
    assert m.should_capture(Pose(0, LAT0, LON0, 5, 0, 0, 0), st, 9)[1] == "below capture altitude"
    assert m.should_capture(p, VehicleStatus(armed=False), 9)[1] == "disarmed"
