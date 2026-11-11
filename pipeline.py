"""
SkyborneMission - the onboard processing pipeline.

    telemetry pose ──┐
                     ├─> should_capture? ─> photo ─> save JPEG + metadata row
    camera frame  ───┘                              ├─> tent detector ─> geolocate ─> tracker
                                                    └─> mosaic (map built during flight)
    finalize():  fused tent GPS ─> annotated map ─> files ─> USB (verified)

The same class runs in every mode (simulation, SITL, replay, real flight);
only the source of frames and poses changes. Nothing here commands the drone.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from camera.image_metadata import MetadataWriter, read_metadata
from communication.telemetry import Pose, VehicleStatus
from config.settings import Settings
from detection.tent_detector import HybridTentDetector
from mapping.camera_model import CameraModel
from mapping.geolocator import Sighting, TargetTracker, locate_pixels, sighting_sigma
from mapping.gps_converter import LocalFrame, point_in_polygon_ne, to_dms_ascii
from mapping.image_mapper import GeoMosaic
from output.export_results import export_to_usb, write_kml, write_manifest
from output.map_renderer import make_closeup, render_map

log = logging.getLogger("pipeline")


class SkyborneMission:
    def __init__(self, cfg: Settings, session_dir: Path, notify=None, frame: LocalFrame | None = None):
        self.cfg = cfg
        self.dir = Path(session_dir)
        for sub in ("images", "thumbs"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.notify = notify or (lambda *a, **k: None)
        self.cam_full = CameraModel.from_config(cfg.camera)
        self.detector = HybridTentDetector(cfg.detection, cfg.resolve(cfg.detection.yolo_model))
        self.tracker = TargetTracker(cfg.geoloc)
        self.meta = MetadataWriter(self.dir / "image_metadata.csv")
        self.det_log = self.dir / "detections.jsonl"
        self.io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="io")
        self.frame: LocalFrame | None = None
        self.mosaic: GeoMosaic | None = None
        self.survey_ne = None
        self.frame_id = 0
        self.last_cap_t = -1e9
        self.last_cap_ne = None
        self.timings: list[float] = []
        self.best_conf_seen = 0.0
        self.finalized = False
        if frame is not None:
            self._init_frame_from(frame)
        sess = self.dir / "session.json"
        if sess.exists():
            info = json.loads(sess.read_text())
            if "frame" in info:
                self._init_frame_from(LocalFrame(info["frame"]["lat0"], info["frame"]["lon0"]))

    # ----------------------------------------------------------- set-up
    def _init_frame_from(self, frame: LocalFrame):
        if self.frame is not None:
            return
        m = self.cfg.mapping
        self.frame = frame
        self.tracker.frame = frame
        self.mosaic = GeoMosaic(frame, m.resolution_m, m.max_canvas_px, m.grow_margin_m, m.refine_alignment,
                                m.max_refine_shift_m, m.min_refine_response, m.exposure_compensation,
                                m.refine_gps_weight)
        if self.cfg.capture.survey_area:
            pts = np.array(self.cfg.capture.survey_area, dtype=float)
            n, e = frame.to_ne(pts[:, 0], pts[:, 1])
            self.survey_ne = np.column_stack((n, e))
        self._write_session(finalized=False)

    def _init_frame(self, pose: Pose):
        if self.frame is None:
            self._init_frame_from(LocalFrame(pose.lat, pose.lon))

    def _write_session(self, finalized: bool, extra: dict | None = None):
        info = {"created": time.time(), "frame": self.frame.to_dict() if self.frame else None,
                "finalized": finalized}
        p = self.dir / "session.json"
        if p.exists():
            old = json.loads(p.read_text())
            info["created"] = old.get("created", info["created"])
        info.update(extra or {})
        p.write_text(json.dumps(info, indent=2))

    def height_for(self, pose: Pose) -> float:
        m = self.cfg.mapping
        if m.use_rangefinder and not math.isnan(pose.rangefinder_m):
            h = pose.rangefinder_m * math.cos(pose.roll) * math.cos(pose.pitch)
        else:
            h = pose.alt - m.ground_elevation_m
        return h + self.cfg.camera.lens_height_on_ground_m

    # --------------------------------------------------------- capture
    def capture_spacing_m(self, height_m: float) -> float:
        _, along = self.cam_full.footprint(height_m)
        return along * (1 - self.cfg.capture.forward_overlap)

    def should_capture(self, pose: Pose | None, status: VehicleStatus, t: float) -> tuple[bool, str]:
        c = self.cfg.capture
        if pose is None:
            return False, "no telemetry"
        if c.require_armed and not status.armed:
            return False, "disarmed"
        if pose.gps_fix < c.min_gps_fix:
            return False, "no 3D GPS fix"
        if pose.alt < c.min_altitude_m:
            return False, "below capture altitude"
        if pose.tilt_deg() > c.max_tilt_deg:
            return False, "tilted"
        self._init_frame(pose)
        n, e = self.frame.to_ne(pose.lat, pose.lon)
        if c.only_in_survey_area and self.survey_ne is not None and \
                not point_in_polygon_ne(n, e, self.survey_ne, c.survey_margin_m):
            return False, "outside survey area"
        dt_ = t - self.last_cap_t
        if dt_ < c.min_interval_s:
            return False, "interval"
        if self.last_cap_ne is not None:
            dist = math.hypot(n - self.last_cap_ne[0], e - self.last_cap_ne[1])
            if dist < self.capture_spacing_m(self.height_for(pose)) and dt_ < c.max_interval_s:
                return False, "spacing"
        self.last_cap_t = t
        self.last_cap_ne = (n, e)
        return True, "ok"

    # ------------------------------------------------------- processing
    def process_frame(self, img: np.ndarray, pose: Pose, t: float | None = None, save: bool = True,
                      battery_pct: float = -1, mode: str = "") -> dict:
        t0 = time.perf_counter()
        self._init_frame(pose)
        fid = self.frame_id
        self.frame_id += 1
        name = f"img_{fid:04d}.jpg"
        h_img, w_img = img.shape[:2]
        cam = self.cam_full if (w_img, h_img) == (self.cam_full.width, self.cam_full.height) \
            else self.cam_full.scaled_to(w_img, h_img)
        H = self.height_for(pose)
        if save:
            self.io.submit(cv2.imwrite, str(self.dir / "images" / name), img,
                           [cv2.IMWRITE_JPEG_QUALITY, self.cfg.camera.jpeg_quality])
        self.meta.write(name, fid, pose, battery_pct, mode)

        # 1) tent detection on the full frame
        gsd = cam.gsd(H)
        dets = self.detector.detect(img, gsd)

        # 2) add to the map at map resolution
        scale = min(1.0, gsd / (self.cfg.mapping.resolution_m * 0.9))
        small = cv2.resize(img, (max(16, int(w_img * scale)), max(16, int(h_img * scale))),
                           interpolation=cv2.INTER_AREA) if scale < 1 else img
        cam_s = cam.scaled_to(small.shape[1], small.shape[0])
        info = self.mosaic.add(small, cam_s, pose, H, fid)
        if self.cfg.output.save_thumbnails:
            self.io.submit(cv2.imwrite, str(self.dir / "thumbs" / name), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        shift = info.get("shift_m", [0.0, 0.0])

        # 3) geolocate every detection
        n0, e0 = self.frame.to_ne(pose.lat, pose.lon)
        rows = []
        for d in dets:
            cx, cy = d.center
            x1, y1, x2, y2 = d.bbox
            pts = np.array([[cx, cy], [x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)
            ne, valid = locate_pixels(cam, self.frame, pose, H, pts)
            if not valid.all():
                continue
            off_nadir = float(cam.off_nadir_deg(pts[:1], pose.roll, pose.pitch, pose.yaw)[0])
            horiz = math.hypot(ne[0, 0] - n0, ne[0, 1] - e0)
            s = Sighting(frame_id=fid, image=name, t=float(pose.t if t is None else t), confidence=float(d.confidence),
                         source=d.source, n=float(ne[0, 0]), e=float(ne[0, 1]),
                         map_n=float(ne[0, 0] + shift[0]), map_e=float(ne[0, 1] + shift[1]),
                         sigma_m=sighting_sigma(self.cfg.geoloc, H, horiz, off_nadir), off_nadir_deg=off_nadir,
                         corners_ne=(ne[1:] + shift).tolist(), bbox=[float(v) for v in d.bbox],
                         altitude=float(H), truncated=d.truncated, size_m=list(d.size_m))
            self.tracker.add(s)
            rows.append(asdict(s) | {"scores": d.scores})
        if rows:
            with self.det_log.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            top = max(r["confidence"] for r in rows)
            if top > self.best_conf_seen + 0.05:
                self.best_conf_seen = top
                lat, lon = self.frame.to_latlon(rows[0]["n"], rows[0]["e"])
                self.notify(f"SKY tent? {top:.2f} {lat:.6f},{lon:.6f}")
        dt_ms = (time.perf_counter() - t0) * 1000
        self.timings.append(dt_ms)
        if fid % 10 == 0:
            self.notify(f"SKY img {fid} map {self.mosaic.coverage_m2():.0f}m2")
        return {"frame_id": fid, "detections": len(rows), "placed": info.get("placed", False),
                "shift_m": shift, "ms": round(dt_ms, 1)}

    # ----------------------------------------------------------- resume
    def resume(self) -> int:
        """Rebuild map + tracker from disk after a reboot (uses the small thumbnails)."""
        meta = self.dir / "image_metadata.csv"
        if not meta.exists() or self.frame is None:
            return 0
        rows = read_metadata(meta)
        for name, fid, pose in rows:
            thumb = cv2.imread(str(self.dir / "thumbs" / name))
            if thumb is None:
                continue
            cam = self.cam_full.scaled_to(thumb.shape[1], thumb.shape[0])
            self.mosaic.add(thumb, cam, pose, self.height_for(pose), fid)
            self.frame_id = max(self.frame_id, fid + 1)
        if self.det_log.exists():
            for line in self.det_log.read_text().splitlines():
                d = json.loads(line)
                d.pop("scores", None)
                self.tracker.add(Sighting(**d))
        log.info("Resumed session: %d photos, %d sightings", len(rows), len(self.tracker.sightings))
        return len(rows)

    # ---------------------------------------------------- map snapping
    def locate_in_map(self, n: float, e: float, radius_m: float):
        """Find the tent as it appears in the finished mosaic near (n, e). Returns (n, e, conf) or None."""
        m = self.mosaic
        if m is None or m.canvas is None:
            return None
        x, y = m.ne_to_px(n, e)
        half = int(radius_m / m.res)
        H, W = m.shape
        x0, y0 = max(0, int(x) - half), max(0, int(y) - half)
        x1, y1 = min(W, int(x) + half), min(H, int(y) + half)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        crop = m.canvas[y0:y1, x0:x1]
        dets = [d for d in self.detector.blob.detect(crop, m.res) if not d.truncated]
        if not dets:
            return None
        def rank(d):
            cx, cy = d.center
            return d.confidence - 0.5 * math.hypot(cx + x0 - x, cy + y0 - y) * m.res / radius_m
        d = max(dets, key=rank)
        if d.confidence < 0.5:
            return None
        cx, cy = d.center
        mn, me = m.px_to_ne(cx + x0, cy + y0)
        return float(mn), float(me), d.confidence

    def _snap_box(self, est):
        """Move the drawn box onto the tent pixels of the mosaic (reported GPS is unchanged)."""
        hit = self.locate_in_map(est.map_n, est.map_e, est.error_95_m + 3.0)
        if hit is not None:
            log.info("Map box snapped by %.2f m onto mosaic tent",
                     math.hypot(hit[0] - est.map_n, hit[1] - est.map_e))
            est.map_n, est.map_e = hit[0], hit[1]

    # --------------------------------------------------------- finalize
    def finalize(self, reason: str = "", usb: bool = True, usb_wait_s: float | None = None,
                 mission_log: Path | None = None) -> dict:
        self.io.shutdown(wait=True)
        self.io = ThreadPoolExecutor(max_workers=1)
        o = self.cfg.output
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.cfg.resolve(o.results_dir) / f"{o.folder_name}_{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        if self.mosaic is None or self.mosaic.canvas is None or self.frame_id < self.cfg.mission.min_images_for_map:
            log.error("Not enough photos for a map (%d)", self.frame_id)
        est = self.tracker.best()
        if est is not None and est.confidence < 0.3:
            log.warning("Best candidate confidence %.2f below report threshold", est.confidence)
            est_report = None
        else:
            est_report = est

        closeup = None
        if est_report is not None:
            self._snap_box(est_report)
            full =cv2.imread(str(self.dir / "images" / est_report.best.image))
            if full is not None:
                closeup = make_closeup(full, est_report.best.bbox, est_report.best.confidence)
                cv2.imwrite(str(out / "tent_closeup.jpg"), closeup, [cv2.IMWRITE_JPEG_QUALITY, 92])

        stats = {"images": self.frame_id, "area_m2": self.mosaic.coverage_m2() if self.mosaic else 0,
                 "detector": self.detector.mode}
        summary = {"reason": reason, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "team": self.cfg.mission.team_name, **stats,
                   "mean_processing_ms": round(float(np.mean(self.timings)), 1) if self.timings else None,
                   "tent_detected": est_report is not None}
        if self.mosaic is not None and self.mosaic.canvas is not None:
            map_img, geo = render_map(self.mosaic, est_report, stats, self.cfg.mission.team_name, closeup, o.max_map_px)
            cv2.imwrite(str(out / "final_map.png"), map_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            cv2.imwrite(str(out / "final_map.jpg"), map_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            # KML overlay (map part only, without the info panel)
            x0, y0, x1, y1 = self.mosaic.valid_bbox(margin_px=int(5 / self.mosaic.res))
            map_h = int(round((y1 - y0) * geo["scale"]))
            cv2.imwrite(str(out / "map_overlay.jpg"), map_img[:map_h], [cv2.IMWRITE_JPEG_QUALITY, 85])
            n_top, e_left = self.mosaic.px_to_ne(x0, y0)
            n_bot, e_right = self.mosaic.px_to_ne(x1, y1)
            north, west = self.frame.to_latlon(n_top, e_left)
            south, east = self.frame.to_latlon(n_bot, e_right)
            bounds = {"north": north, "south": south, "east": east, "west": west}
            write_kml(out / "skyborne_map.kml", "map_overlay.jpg", bounds, est_report)
            summary["map_bounds"] = bounds
            summary["map_resolution_m"] = geo["res_m"]

        # text + json reports
        lines = [f"{self.cfg.mission.team_name}", f"Generated (UTC): {summary['generated_utc']}", ""]
        if est_report is not None:
            e = est_report
            lines += ["WHITE TENT DETECTED", f"Latitude : {e.lat:.7f}  ({to_dms_ascii(e.lat, True)})",
                      f"Longitude: {e.lon:.7f}  ({to_dms_ascii(e.lon, False)})",
                      f"Estimated accuracy: +/- {e.error_95_m:.1f} m (95 %)",
                      f"Detection confidence: {e.confidence:.2f}", f"Sightings fused: {e.sightings}",
                      f"Best photo: {e.best.image}  bbox {[round(v) for v in e.best.bbox]}"]
            summary["tent"] = {"lat": e.lat, "lon": e.lon, "confidence": e.confidence,
                               "error_95_m": e.error_95_m, "sightings": e.sightings}
        else:
            lines += ["WHITE TENT NOT DETECTED"]
        lines += ["", "Coordinates estimated from camera geometry, GPS, altitude and attitude",
                  "(flat-ground model). Datum: WGS-84."]
        (out / "tent_coordinates.txt").write_text("\n".join(lines) + "\n")
        cands = self.tracker.candidates()
        (out / "detections.json").write_text(json.dumps({
            "detected": est_report is not None,
            "estimate": est_report.to_dict() if est_report else None,
            "candidates": [{"lat": c.lat, "lon": c.lon, "confidence": c.confidence, "sightings": c.sightings,
                            "error_95_m": c.error_95_m} for c in cands[:10]],
        }, indent=2))
        shutil.copy(self.dir / "image_metadata.csv", out / "image_metadata.csv")
        if mission_log and Path(mission_log).exists():
            shutil.copy(mission_log, out / "mission_log.csv")
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        write_manifest(out)
        result = {"output_dir": str(out), "summary": summary, "estimate": est_report}
        log.info("Results written to %s", out)
        if usb:
            result["usb"] = export_to_usb(out, o.usb_path, o.usb_wait_s if usb_wait_s is None else usb_wait_s,
                                          self.notify)
        self._write_session(finalized=True, extra={"output": str(out)})
        self.finalized = True
        return result


class MissionClock:
    """Tracks the mission time limit (starts at first arm or at program start)."""

    def __init__(self, cfg: Settings, now: float):
        self.cfg = cfg.mission
        self.t0 = now if self.cfg.clock_start == "start" else None

    def update(self, status: VehicleStatus, now: float):
        if self.t0 is None and status.armed:
            self.t0 = now

    def remaining(self, now: float) -> float:
        return float("inf") if self.t0 is None else self.cfg.time_limit_s - (now - self.t0)

    def must_finalize(self, now: float) -> bool:
        return self.remaining(now) <= self.cfg.finalize_margin_s
