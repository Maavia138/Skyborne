"""
Generate an ArduPilot mission file (Mission Planner / QGC "WPL 110" format):

    take-off -> N laps of the lap course (DO_JUMP) -> lawn-mower survey of the
    search area (line spacing from the camera footprint) -> RTL

and `config/user_settings.json` survey polygon for the onboard capture gate.

    python tools/make_mission.py --home 33.6844,73.0479 \
        --laps-file laps.csv --laps 5 \
        --survey-center 33.6850,73.0479 --survey-size 140x120 --survey-heading 0 \
        --alt 30 --speed 5 --out missions/neo26.waypoints

laps.csv: one "lat,lon" per line (the lap course waypoints).
Load the .waypoints file in Mission Planner (Plan -> Load WP File), CHECK IT
VISUALLY, then write it to the flight controller. Always fly it in SITL first.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CONFIG_DIR, Settings  # noqa: E402
from mapping.camera_model import CameraModel  # noqa: E402
from mapping.gps_converter import LocalFrame  # noqa: E402

CMD_WAYPOINT, CMD_RTL, CMD_TAKEOFF, CMD_JUMP, CMD_SPEED = 16, 20, 22, 177, 178
FRAME_REL = 3   # MAV_FRAME_GLOBAL_RELATIVE_ALT


def survey_lines(center, size_ne, heading_deg, spacing, overshoot=8.0):
    frame = LocalFrame(*center)
    sn, se = size_ne
    h = math.radians(heading_deg)
    u = (math.cos(h), math.sin(h))          # along-line direction
    v = (-math.sin(h), math.cos(h))         # across-line direction
    n_lines = int(math.ceil(se / spacing)) + 1
    pts = []
    for i in range(n_lines):
        off = -se / 2 + i * se / (n_lines - 1)
        a, b = (-sn / 2 - overshoot, sn / 2 + overshoot)
        if i % 2:
            a, b = b, a
        for s in (a, b):
            n = s * u[0] + off * v[0]
            e = s * u[1] + off * v[1]
            pts.append(frame.to_latlon(n, e))
    corners = []
    for cs, cv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        n = cs * sn / 2 * u[0] + cv * se / 2 * v[0]
        e = cs * sn / 2 * u[1] + cv * se / 2 * v[1]
        corners.append(list(frame.to_latlon(n, e)))
    return pts, corners, n_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True, help="lat,lon")
    ap.add_argument("--laps-file")
    ap.add_argument("--laps", type=int, default=5)
    ap.add_argument("--survey-center", required=True, help="lat,lon")
    ap.add_argument("--survey-size", default="140x120", help="NxE metres (along x across)")
    ap.add_argument("--survey-heading", type=float, default=0.0, help="line direction, deg from north")
    ap.add_argument("--alt", type=float, default=30.0)
    ap.add_argument("--lap-alt", type=float, default=None)
    ap.add_argument("--speed", type=float, default=5.0, help="survey speed m/s")
    ap.add_argument("--lap-speed", type=float, default=8.0)
    ap.add_argument("--out", default="missions/skyborne.waypoints")
    ap.add_argument("--write-settings", action="store_true", help="store survey polygon in user_settings.json")
    a = ap.parse_args()

    cfg = Settings.load()
    home = tuple(float(x) for x in a.home.split(","))
    center = tuple(float(x) for x in a.survey_center.split(","))
    size = tuple(float(x) for x in a.survey_size.lower().split("x"))
    cam = CameraModel.from_config(cfg.camera)
    fw, fl = cam.footprint(a.alt)
    spacing = fw * (1 - cfg.capture.side_overlap)
    lap_alt = a.lap_alt or a.alt

    items = [(0, 0, 0, home[0], home[1], 0)]                           # home (index 0)
    items.append((FRAME_REL, CMD_TAKEOFF, (0, 0, 0, 0), home[0], home[1], lap_alt))
    lap_start = None
    if a.laps_file:
        laps = [tuple(map(float, l.split(",")[:2])) for l in Path(a.laps_file).read_text().split() if l.strip()]
        items.append((FRAME_REL, CMD_SPEED, (1, a.lap_speed, -1, 0), 0, 0, 0))
        lap_start = len(items)
        for lat, lon in laps:
            items.append((FRAME_REL, CMD_WAYPOINT, (0, 0, 0, 0), lat, lon, lap_alt))
        if a.laps > 1:
            items.append((FRAME_REL, CMD_JUMP, (lap_start, a.laps - 1, 0, 0), 0, 0, 0))
    pts, corners, n_lines = survey_lines(center, size, a.survey_heading, spacing)
    items.append((FRAME_REL, CMD_SPEED, (1, a.speed, -1, 0), 0, 0, 0))
    for lat, lon in pts:
        items.append((FRAME_REL, CMD_WAYPOINT, (0, 0, 0, 0), lat, lon, a.alt))
    items.append((FRAME_REL, CMD_RTL, (0, 0, 0, 0), 0, 0, 0))

    lines = ["QGC WPL 110"]
    for i, it in enumerate(items):
        if i == 0:
            lines.append(f"0\t1\t0\t16\t0\t0\t0\t0\t{home[0]:.8f}\t{home[1]:.8f}\t0.000000\t1")
            continue
        frame, cmd, p, lat, lon, alt = it
        lines.append(f"{i}\t0\t{frame}\t{cmd}\t{p[0]}\t{p[1]}\t{p[2]}\t{p[3]}\t{lat:.8f}\t{lon:.8f}\t{alt:.6f}\t1")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    length = 2 * (size[0] + 16) * n_lines / 2 + size[1]
    print(f"Mission written: {out}  ({len(items) - 1} items)")
    print(f"Survey: {n_lines} lines, spacing {spacing:.1f} m, footprint {fw:.1f}x{fl:.1f} m at {a.alt:.0f} m, "
          f"~{length:.0f} m of survey path ~ {length / a.speed / 60:.1f} min at {a.speed} m/s")
    print(f"Photo every {fl * (1 - cfg.capture.forward_overlap):.1f} m "
          f"({fl * (1 - cfg.capture.forward_overlap) / a.speed:.1f} s at {a.speed} m/s)")
    if a.write_settings:
        p = CONFIG_DIR / "user_settings.json"
        data = json.loads(p.read_text()) if p.exists() else {}
        data.setdefault("capture", {})["survey_area"] = corners
        p.write_text(json.dumps(data, indent=2))
        print(f"Survey polygon saved to {p}")


if __name__ == "__main__":
    main()
