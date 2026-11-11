"""
Skyborne NEO'26 - onboard perception & mapping software.

    python main.py sim                       # full mission in simulation (laptop, no hardware)
    python main.py sitl                      # ArduPilot SITL telemetry + synthetic camera
    python main.py flight                    # Raspberry Pi 5 + Pixhawk (TELEM2) + Pi Camera v2
    python main.py replay --session DIR      # re-process a recorded flight
    python main.py detect --image a.jpg --alt 30   # test the detector on one photo

The program never commands the aircraft. ArduPilot flies the laps/survey and
owns every failsafe; this program only reads telemetry, takes photos, builds
the map, finds the tent and writes the USB stick.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import signal
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import Settings, describe_camera  # noqa: E402
from mapping.camera_model import CameraModel  # noqa: E402
from mapping.gps_converter import haversine_m  # noqa: E402
from pipeline import MissionClock, SkyborneMission  # noqa: E402

log = logging.getLogger("skyborne")


# --------------------------------------------------------------------- utils
def setup_logging(session: Path | None, verbose=False):
    handlers = [logging.StreamHandler(sys.stdout)]
    if session:
        handlers.append(logging.FileHandler(session / "skyborne.log", encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)-7s %(name)-9s %(message)s", datefmt="%H:%M:%S")


def new_session(cfg: Settings, prefix: str) -> Path:
    d = cfg.resolve(cfg.output.data_dir) / "sessions" / f"{prefix}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True, exist_ok=True)
    (cfg.resolve(cfg.output.data_dir) / "current_session.txt").write_text(str(d))
    return d


def resumable_session(cfg: Settings) -> Path | None:
    ptr = cfg.resolve(cfg.output.data_dir) / "current_session.txt"
    if not ptr.exists():
        return None
    d = Path(ptr.read_text().strip())
    info = d / "session.json"
    if not info.exists():
        return None
    s = json.loads(info.read_text())
    if s.get("finalized") or time.time() - s.get("created", 0) > cfg.mission.resume_max_age_s:
        return None
    return d


# ---------------------------------------------------------------- simulation
def run_sim(cfg: Settings, args) -> dict:
    from simulation.sim_camera import SimCamera
    from simulation.sim_flight import SimFlight
    from simulation.world import SyntheticWorld

    if args.seed is not None:
        cfg.sim.seed = args.seed
    session = new_session(cfg, "sim")
    setup_logging(session, args.verbose)
    log.info("Building synthetic world (seed %d) ...", cfg.sim.seed)
    t0 = time.time()
    world = SyntheticWorld(cfg.sim)
    full = CameraModel.from_config(cfg.camera)
    sc = cfg.sim.render_scale
    simcam = SimCamera(world, full.scaled_to(int(full.width * sc), int(full.height * sc)),
                       cfg.camera.lens_height_on_ground_m, cfg.sim.seed)
    flight = SimFlight(cfg, world, full)
    cfg.capture.survey_area = world.survey_polygon_latlon()
    log.info("World ready in %.1f s. %s", time.time() - t0, describe_camera(cfg.camera, cfg.sim.altitude_m))
    log.info("Survey %gx%g m, %d lines, spacing %.1f m, flight %.0f s",
             *cfg.sim.survey_size_m, flight.lines, flight.spacing, flight.duration)

    mission = SkyborneMission(cfg, session, notify=lambda m, *a: log.info("[GCS] %s", m))
    t, dt_ = 0.0, 0.1
    while t < flight.duration:
        pose = flight.measured_pose(t)
        ok, _ = mission.should_capture(pose, flight.status(t), t)
        if ok:
            img = simcam.render(flight.true_pose(t))
            r = mission.process_frame(img, pose, t, battery_pct=flight.status(t).battery_pct, mode="AUTO")
            log.debug("frame %s", r)
        t += dt_
    log.info("Flight finished: %d photos, mean processing %.0f ms/photo (laptop)", mission.frame_id,
             sum(mission.timings) / max(1, len(mission.timings)))

    if args.usb_dir:
        cfg.output.usb_path = args.usb_dir
        Path(args.usb_dir).mkdir(parents=True, exist_ok=True)
    res = mission.finalize("simulation complete", usb=bool(args.usb_dir), usb_wait_s=0)

    # ------------------------------------------------- score against truth
    truth = world.tent
    est = res["estimate"]
    report = {"truth": {"lat": truth["lat"], "lon": truth["lon"]}, "photos": mission.frame_id,
              "seed": cfg.sim.seed}
    if est is not None:
        err = haversine_m(truth["lat"], truth["lon"], est.lat, est.lon)
        tn, te = mission.frame.to_ne(truth["lat"], truth["lon"])
        map_err = math.hypot(est.map_n - tn, est.map_e - te)
        # where does the tent actually appear in the mosaic? (absolute map accuracy + box placement)
        hit = mission.locate_in_map(tn, te, 8.0)
        georef = math.hypot(hit[0] - tn, hit[1] - te) if hit else None
        box_vs_tent = math.hypot(hit[0] - est.map_n, hit[1] - est.map_e) if hit else None
        report.update(detected=True, gps_error_m=round(err, 2),
                      map_georef_error_m=None if georef is None else round(georef, 2),
                      box_vs_tent_in_map_m=None if box_vs_tent is None else round(box_vs_tent, 2),
                      claimed_error_95_m=est.error_95_m, confidence=est.confidence, sightings=est.sightings,
                      within_claim=err <= est.error_95_m)
        log.info("TRUTH %.7f, %.7f | ESTIMATE %.7f, %.7f | GPS error %.2f m (claimed 95%%: %.2f m) | "
                 "map georef error %s m | box vs tent in map %s m", truth["lat"], truth["lon"], est.lat, est.lon,
                 err, est.error_95_m, report["map_georef_error_m"], report["box_vs_tent_in_map_m"])
    else:
        report["detected"] = False
        log.warning("Tent NOT detected")
    false_pos = [c for c in mission.tracker.candidates()[1:] if c.confidence >= 0.3]
    report["other_candidates_over_0.3"] = len(false_pos)
    Path(res["output_dir"], "sim_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------- live (SITL/flight)
def run_live(cfg: Settings, args, sitl: bool):
    from communication.mavlink_reader import MavlinkTelemetry

    resume_dir = None if args.new_session else resumable_session(cfg)
    session = resume_dir or new_session(cfg, "sitl" if sitl else "flight")
    setup_logging(session, args.verbose)
    conn = args.connect or (cfg.mavlink.sitl_connection if sitl else cfg.mavlink.connection)
    m = cfg.mavlink
    tel = MavlinkTelemetry(conn, args.baud or m.baud, m.stream_rate_hz, session / "mission_log.csv", m.log_rate_hz,
                           m.send_heartbeat, m.send_statustext, m.heartbeat_timeout_s).start()
    log.info("Waiting for flight controller on %s ...", conn)
    while not tel.wait_connected(10):
        log.info("... still waiting for heartbeat + position")

    if sitl:
        from simulation.sim_camera import SitlSimCamera
        camera = SitlSimCamera(cfg, tel).start()
        cfg.capture.survey_area = camera.world.survey_polygon_latlon()
    else:
        from camera.camera_capture import create_camera
        camera = create_camera(cfg.camera).start()

    mission = SkyborneMission(cfg, session, notify=tel.send_status)
    if resume_dir:
        n = mission.resume()
        tel.send_status(f"SKY resumed {n} photos")
    tel.send_status("SKY onboard ready")
    clock = MissionClock(cfg, time.monotonic())
    trigger = cfg.resolve(cfg.output.data_dir) / "FINALIZE"
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig)

    was_armed, reason, last_report = False, "", 0.0
    try:
        while not stop["flag"]:
            now = time.monotonic()
            st = tel.status()
            clock.update(st, now)
            pose = tel.latest()
            ok, why = mission.should_capture(pose, st, now)
            if ok:
                frame, t_exp = camera.capture()
                p = tel.pose_at(t_exp) or pose
                r = mission.process_frame(frame, p, t_exp, battery_pct=st.battery_pct, mode=st.mode)
                log.info("photo %d  alt %.1f m  dets %d  %.0f ms", r["frame_id"], p.alt, r["detections"], r["ms"])
            if st.armed:
                was_armed = True
            if now - last_report > 10:
                last_report = now
                log.info("status %s armed=%d batt %.0f%% photos %d (%s) time left %.0f s", st.mode, st.armed,
                         st.battery_pct, mission.frame_id, why, clock.remaining(now))
            enough = mission.frame_id >= cfg.mission.min_images_for_map
            if enough and cfg.mission.finalize_on_disarm and was_armed and st.connected and not st.armed:
                reason = "landed and disarmed"
            elif enough and clock.must_finalize(now):
                reason = "mission time limit approaching"
            elif trigger.exists():
                trigger.unlink()
                reason = "manual trigger file"
            if reason:
                break
            time.sleep(0.02)
        if not reason:
            reason = "operator stop"
        log.info("Finalizing: %s", reason)
        tel.send_status("SKY building final map")
        res = mission.finalize(reason, usb=True, mission_log=session / "mission_log.csv")
        est = res["estimate"]
        if est:
            tel.send_status(f"SKY TENT {est.lat:.6f},{est.lon:.6f}")
        log.info("USB: %s", res.get("usb"))
        time.sleep(2)  # let the last STATUSTEXT go out
    finally:
        camera.stop()
        tel.stop()


# --------------------------------------------------------------------- replay
def run_replay(cfg: Settings, args):
    from camera.image_metadata import read_metadata
    src = Path(args.session)
    meta = Path(args.metadata) if args.metadata else src / "image_metadata.csv"
    img_dir = Path(args.images) if args.images else src / "images"
    session = new_session(cfg, "replay")
    setup_logging(session, args.verbose)
    mission = SkyborneMission(cfg, session)
    rows = read_metadata(meta)
    for name, _, pose in rows:
        img = cv2.imread(str(img_dir / name))
        if img is None:
            log.warning("missing %s", name)
            continue
        r = mission.process_frame(img, pose, pose.t)
        log.info("%s -> %s", name, r)
    res = mission.finalize("replay", usb=False)
    log.info("Output: %s", res["output_dir"])


def run_detect(cfg: Settings, args):
    from detection.tent_detector import HybridTentDetector, draw_detections
    setup_logging(None, args.verbose)
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"cannot read {args.image}")
    cam = CameraModel.from_config(cfg.camera).scaled_to(img.shape[1], img.shape[0])
    gsd = args.gsd if args.gsd else cam.gsd(args.alt)
    det = HybridTentDetector(cfg.detection, cfg.resolve(cfg.detection.yolo_model))
    t0 = time.perf_counter()
    dets = det.detect(img, gsd)
    log.info("%d detections in %.0f ms (GSD %.2f cm/px, mode %s)", len(dets), (time.perf_counter() - t0) * 1000,
             gsd * 100, det.mode)
    for d in dets:
        print(json.dumps(d.to_dict()))
    out = args.out or str(Path(args.image).with_name(Path(args.image).stem + "_det.jpg"))
    cv2.imwrite(out, draw_detections(img, dets))
    log.info("annotated image: %s", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON file with setting overrides")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sim", help="full simulated mission")
    s.add_argument("--seed", type=int)
    s.add_argument("--usb-dir", help="folder that acts as the USB stick")
    for name in ("sitl", "flight"):
        p = sub.add_parser(name)
        p.add_argument("--connect", help="MAVLink connection string")
        p.add_argument("--baud", type=int)
        p.add_argument("--new-session", action="store_true", help="do not resume an unfinished session")
    r = sub.add_parser("replay")
    r.add_argument("--session", required=True)
    r.add_argument("--images")
    r.add_argument("--metadata")
    d = sub.add_parser("detect")
    d.add_argument("--image", required=True)
    d.add_argument("--alt", type=float, default=30.0)
    d.add_argument("--gsd", type=float, help="metres per pixel (overrides --alt)")
    d.add_argument("--out")
    args = ap.parse_args()
    cfg = Settings.load(args.config)
    if args.mode == "sim":
        run_sim(cfg, args)
    elif args.mode in ("sitl", "flight"):
        run_live(cfg, args, sitl=args.mode == "sitl")
    elif args.mode == "replay":
        run_replay(cfg, args)
    elif args.mode == "detect":
        run_detect(cfg, args)


if __name__ == "__main__":
    main()
