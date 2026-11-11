"""
Central configuration for the Skyborne onboard software.

Every tunable number lives here. Defaults are chosen for:
  * Raspberry Pi Camera Module v2 (Sony IMX219, 3280x2464, f = 3.04 mm)
  * survey altitude ~30 m, ground speed 3-5 m/s
  * ArduPilot flight controller over MAVLink (TELEM2)

Override anything without touching code by creating `config/user_settings.json`
(or passing --config my.json). Only the keys you want to change are needed, e.g.

    {"capture": {"forward_overlap": 0.8}, "mission": {"time_limit_s": 900}}

If `config/camera_calibration.json` exists (made by tools/calibrate_camera.py),
its intrinsics replace the nominal datasheet values automatically.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass
class CameraConfig:
    backend: str = "picamera2"          # picamera2 | opencv | sim
    width: int = 3280                   # capture resolution (full IMX219 sensor)
    height: int = 2464
    sensor_width_px: int = 3280         # native sensor width (for focal-length scaling)
    focal_length_mm: float = 3.04       # Pi Camera v2 lens
    pixel_size_um: float = 1.12         # IMX219 pixel pitch
    fx: float | None = None             # pixels; None -> from focal length & pixel size
    fy: float | None = None
    cx: float | None = None             # principal point; None -> image centre
    cy: float | None = None
    dist_coeffs: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    # Camera mounting relative to the body (0,0,0 = straight down, image top = nose)
    mount_roll_deg: float = 0.0
    mount_pitch_deg: float = 0.0
    mount_yaw_deg: float = 0.0
    lens_height_on_ground_m: float = 0.25  # lens height when landed (relative alt = 0 there)
    exposure_us: int = 1000             # 1/1000 s against motion blur and vibration
    analogue_gain: float | None = None  # None -> let AGC pick the gain at fixed shutter
    jpeg_quality: int = 90
    capture_latency_s: float = 0.03     # fallback if sensor timestamp unavailable
    opencv_index: int = 0


@dataclass
class MavlinkConfig:
    # Pi 5: GPIO14/15 UART is /dev/ttyAMA0 after `dtparam=uart0=on` in config.txt
    connection: str = "/dev/ttyAMA0"
    baud: int = 921600                  # must match SERIAL2_BAUD on the flight controller
    sitl_connection: str = "udpin:0.0.0.0:14551"
    stream_rate_hz: float = 10.0
    heartbeat_timeout_s: float = 30.0
    send_heartbeat: bool = True         # identify as onboard computer (read-only)
    send_statustext: bool = True        # progress messages shown in Mission Planner
    log_rate_hz: float = 2.0


@dataclass
class CaptureConfig:
    min_altitude_m: float = 15.0        # no photos during take-off / landing
    max_tilt_deg: float = 20.0          # skip strongly banked frames (turns)
    forward_overlap: float = 0.75       # along-track overlap between photos
    side_overlap: float = 0.40          # used by mission planner tool & simulator
    min_interval_s: float = 0.8         # processing budget on the Pi
    max_interval_s: float = 3.0         # always capture at least this often in area
    require_armed: bool = True
    only_in_survey_area: bool = True
    survey_area: list | None = None     # [[lat, lon], ...] polygon; None -> anywhere
    survey_margin_m: float = 15.0
    min_gps_fix: int = 3                # 3 = 3D fix


@dataclass
class DetectionConfig:
    proc_width: int = 1640              # detection working width (half res)
    # White-blob stage (HSV)
    white_s_max: int = 55               # max saturation for "white" (0-255)
    white_v_min: int = 165              # absolute minimum brightness
    adaptive_v_frac: float = 0.55       # thr = med + frac*(p99.9 - med)
    tophat_min: int = 20                # must be brighter than surroundings
    tent_min_area_m2: float = 2.5
    tent_max_area_m2: float = 22.0
    tent_min_side_m: float = 1.2
    tent_max_side_m: float = 6.0
    tent_max_aspect: float = 2.2
    max_dark_inside: float = 0.12       # cars have dark windows inside the white body
    min_confidence: float = 0.35
    # Neural stage (YOLO, ONNX)
    use_yolo: str = "auto"              # auto | on | off
    yolo_model: str = "models/tent_yolo.onnx"
    yolo_imgsz: int = 640
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45
    yolo_class_names: list = field(default_factory=lambda: ["white_tent"])
    yolo_tent_class: int = 0
    yolo_tiles: int = 1                 # 2 -> 2x2 tiles (higher altitude / small tents)
    hsv_only_weight: float = 0.6        # confidence scale for blob-only candidates when YOLO is on


@dataclass
class MappingConfig:
    resolution_m: float = 0.05          # map ground sample distance (5 cm/pixel)
    max_canvas_px: int = 16000
    grow_margin_m: float = 40.0
    refine_alignment: bool = True       # image-based correction of GPS placement
    max_refine_shift_m: float = 3.0
    min_refine_response: float = 0.08
    refine_gps_weight: float = 0.2      # 0 = trust images fully (drifts), 1 = GPS only (seams)
    exposure_compensation: bool = True
    ground_elevation_m: float = 0.0     # survey ground height relative to home
    use_rangefinder: bool = False       # use DISTANCE_SENSOR (down) for height if fitted


@dataclass
class GeolocationConfig:
    cluster_radius_m: float = 6.0
    gps_sigma_m: float = 1.5            # GPS bias (does not average out)
    gps_noise_m: float = 0.3            # GPS jitter (averages out)
    att_sigma_deg: float = 1.0          # roll/pitch error incl. camera mount
    yaw_sigma_deg: float = 2.5          # compass heading error
    alt_sigma_m: float = 0.8


@dataclass
class MissionConfig:
    team_name: str = "Team Skyborne - NEO'26"
    time_limit_s: float = 1200.0        # ASK ORGANISERS: mission time limit
    finalize_margin_s: float = 150.0    # finish the map this long before the limit
    clock_start: str = "arm"            # arm | start
    min_images_for_map: int = 3
    finalize_on_disarm: bool = True
    resume_max_age_s: float = 3 * 3600  # resume an unfinished session after a reboot


@dataclass
class OutputConfig:
    data_dir: str = "data"
    results_dir: str = "results"
    usb_path: str | None = None         # None -> auto-detect removable drive
    usb_wait_s: float = 900.0           # keep waiting for a USB stick after landing
    folder_name: str = "SKYBORNE_OUTPUT"
    max_map_px: int = 12000             # longest side of the exported map
    save_thumbnails: bool = True        # small copies for fast resume after reboot
    thumb_width: int = 820


@dataclass
class SimConfig:
    home_lat: float = 33.6844
    home_lon: float = 73.0479
    survey_size_m: list = field(default_factory=lambda: [140.0, 120.0])  # [north, east]
    world_size_m: list = field(default_factory=lambda: [240.0, 220.0])
    world_res_m: float = 0.05
    altitude_m: float = 30.0
    speed_mps: float = 5.0
    render_scale: float = 0.5           # simulated camera at half resolution (speed)
    seed: int = 7
    tent_ne: list | None = None         # [north, east] metres from survey centre; None = random
    tent_size_m: list = field(default_factory=lambda: [2.4, 2.1])
    decoys: bool = True
    gps_bias_m: float = 1.2
    gps_noise_m: float = 0.25
    att_noise_deg: float = 0.4
    yaw_bias_deg: float = 1.5
    alt_bias_m: float = 0.4
    wind_tilt_deg: float = 3.0


@dataclass
class Settings:
    camera: CameraConfig = field(default_factory=CameraConfig)
    mavlink: MavlinkConfig = field(default_factory=MavlinkConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    geoloc: GeolocationConfig = field(default_factory=GeolocationConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    # ---------------------------------------------------------------- helpers
    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: str | Path | None = None) -> "Settings":
        s = Settings()
        user = Path(path) if path else CONFIG_DIR / "user_settings.json"
        if user.exists():
            _merge(s, json.loads(user.read_text()))
        calib = CONFIG_DIR / "camera_calibration.json"
        if calib.exists():
            _apply_calibration(s.camera, json.loads(calib.read_text()))
        return s


def _merge(obj: Any, data: dict) -> None:
    names = {f.name for f in fields(obj)}
    for k, v in data.items():
        if k not in names:
            raise KeyError(f"Unknown setting '{k}' in {type(obj).__name__}")
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(obj, k, v)


def _apply_calibration(cam: CameraConfig, c: dict) -> None:
    """Calibration was done at c['width'] x c['height']; rescale to capture size."""
    sx = cam.width / c["width"]
    sy = cam.height / c["height"]
    cam.fx, cam.fy = c["fx"] * sx, c["fy"] * sy
    cam.cx, cam.cy = c["cx"] * sx, c["cy"] * sy
    cam.dist_coeffs = list(c.get("dist_coeffs", cam.dist_coeffs))


def nominal_fx(cam: CameraConfig) -> float:
    fx_native = cam.focal_length_mm / (cam.pixel_size_um * 1e-3)
    return fx_native * cam.width / cam.sensor_width_px


def describe_camera(cam: CameraConfig, altitude_m: float) -> str:
    fx = cam.fx or nominal_fx(cam)
    gsd = altitude_m / fx
    hfov = math.degrees(2 * math.atan(cam.width / (2 * fx)))
    return (f"GSD {gsd * 100:.2f} cm/px at {altitude_m:.0f} m, footprint "
            f"{cam.width * gsd:.1f} x {cam.height * gsd:.1f} m, HFOV {hfov:.1f} deg")
