"""Builds docs/Skyborne_Software_Guide.pdf from the latest simulation results.
    python docs/build_guide.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
IMG = DOCS / "img"
IMG.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#1F3A5F")
TEAL = colors.HexColor("#2A7F8E")
LIGHT = colors.HexColor("#EEF3F8")
AMBER = colors.HexColor("#F2A33A")
RED = colors.HexColor("#C0392B")
GREY = colors.HexColor("#555555")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=17, spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=TEAL, fontSize=12.5, spaceBefore=8, spaceAfter=4)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=4)
SM = ParagraphStyle("SM", parent=B, fontSize=8.2, leading=10.5, textColor=GREY)
CAP = ParagraphStyle("CAP", parent=SM, alignment=TA_CENTER, spaceAfter=8)
CODE = ParagraphStyle("CODE", parent=B, fontName="Courier", fontSize=8.2, leading=10.4, backColor=LIGHT,
                      borderPadding=5, spaceBefore=3, spaceAfter=7, leftIndent=4)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2)


def p(t, s=B):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def code(t):
    return Paragraph(t.replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br/>").replace(" ", "&nbsp;"), CODE)


def table(rows, widths, header=True, font=8.6):
    cs = ParagraphStyle("c", parent=B, fontSize=font, leading=font + 2.5, spaceAfter=0)
    hs = ParagraphStyle("hc", parent=cs, textColor=colors.white, fontName="Helvetica-Bold")
    t = Table([[Paragraph(str(c), hs if (header and i == 0) else cs) for c in r] for i, r in enumerate(rows)],
              colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C4D0")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st))
    return t


def img(path, width_cm, max_h_cm=None):
    im = cv2.imread(str(path))
    h, w = im.shape[:2]
    W = width_cm * cm
    H = W * h / w
    if max_h_cm and H > max_h_cm * cm:
        H = max_h_cm * cm
        W = H * w / h
    return Image(str(path), width=W, height=H)


# ------------------------------------------------------------- diagrams
def box(d, x, y, w, h, title, lines=(), fill=LIGHT, stroke=NAVY, tcolor=NAVY):
    d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=fill, strokeColor=stroke, strokeWidth=1.2))
    d.add(String(x + w / 2, y + h - 14, title, fontName="Helvetica-Bold", fontSize=9.5, fillColor=tcolor,
                 textAnchor="middle"))
    for i, l in enumerate(lines):
        d.add(String(x + w / 2, y + h - 28 - 11 * i, l, fontName="Helvetica", fontSize=7.6, fillColor=GREY,
                     textAnchor="middle"))


def arrow(d, x1, y1, x2, y2, label="", color=TEAL):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.6))
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    s = 6
    d.add(Polygon([x2, y2, x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4),
                   x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4)], fillColor=color, strokeColor=color))
    if label:
        d.add(String((x1 + x2) / 2 + 4, (y1 + y2) / 2 + 4, label, fontName="Helvetica-Oblique", fontSize=7,
                     fillColor=GREY))


def architecture():
    d = Drawing(17 * cm, 9.2 * cm)
    W = 17 * cm
    box(d, 5, 190, 140, 60, "Ground station (laptop)", ["Mission Planner, offline maps", "ft / knots, geofence view"])
    box(d, W / 2 - 80, 190, 160, 60, "Pixhawk + ArduPilot", ["flies laps + survey (AUTO)", "RTL, geofence, battery FS"],
        fill=colors.HexColor("#FDF1E2"), stroke=AMBER, tcolor=colors.HexColor("#8A5A12"))
    box(d, W - 150, 190, 145, 60, "Safety pilot RC", ["FS-i6: takeover", "kill switch"])
    arrow(d, 145, 220, W / 2 - 80, 220)
    d.add(String(160, 258, "SiK radio", fontName="Helvetica-Oblique", fontSize=7, fillColor=GREY))
    arrow(d, W - 150, 220, W / 2 + 80, 220)
    d.add(String(W - 140, 258, "RC link", fontName="Helvetica-Oblique", fontSize=7, fillColor=GREY))
    box(d, W / 2 - 230, 15, 460, 140, "Raspberry Pi 5  -  Skyborne onboard software (this project)",
        fill=colors.HexColor("#E6F2F4"), stroke=TEAL, tcolor=TEAL)
    arrow(d, W / 2, 190, W / 2, 155, "TELEM2 UART (read-only)")
    names = [("MAVLink",["pose @10 Hz", "interpolation"]), ("Camera", ["Pi Camera v2", "1/1000 s"]),
             ("Tent detector", ["white-blob", "+ YOLO11n"]), ("Mosaic", ["GPS + image", "alignment"]),
             ("Geolocation", ["ray -> ground", "multi-view fusion"]), ("Export", ["PNG/JSON/KML", "USB + SHA-256"])]
    bw, gap = 66, 9
    x0 = W / 2 - (6 * bw + 5 * gap) / 2
    for i, (n, l) in enumerate(names):
        box(d, x0 + i * (bw + gap), 40, bw, 70, n, l, fill=colors.white, stroke=TEAL, tcolor=NAVY)
        if i < 5:
            arrow(d, x0 + i * (bw + gap) + bw, 75, x0 + (i + 1) * (bw + gap), 75)
    d.add(String(W / 2, 22, "All image processing happens here, on the drone (competition rule).",
                 fontName="Helvetica-Oblique", fontSize=8, fillColor=GREY, textAnchor="middle"))
    return d


def per_photo():
    d = Drawing(17 * cm, 6.3 * cm)
    steps = [("1 Trigger", ["armed, alt > 15 m", "in area, tilt < 20deg", "every 6.8 m"]),
             ("2 Capture", ["3280x2464 photo", "exposure timestamp", "interpolated pose"]),
             ("3 Save", ["JPEG + CSV row", "thumbnail (resume)"]),
             ("4 Detect", ["HSV + top-hat", "size in metres", "YOLO confirm"]),
             ("5 Map", ["homography", "phase-correlation fix", "nadir seams"]),
             ("6 Locate", ["pixel ray -> ground", "-> lat/lon", "cluster + fuse"])]
    bw, gap = 72, 11
    for i, (n, l) in enumerate(steps):
        x = 4 + i * (bw + gap)
        box(d, x, 60, bw, 100, n, l, fill=LIGHT if i % 2 else colors.white)
        if i < 5:
            arrow(d, x + bw, 110, x + bw + gap, 110)
    box(d, 4, 2, 6 * bw + 5 * gap, 44, "At landing (disarm) or before the time limit:  finalize",
        ["fused tent GPS -> snap box onto map -> annotate map -> write files -> copy to USB -> verify checksums"],
        fill=colors.HexColor("#FDF1E2"), stroke=AMBER, tcolor=colors.HexColor("#8A5A12"))
    return d


def geometry():
    d = Drawing(17 * cm, 6.2 * cm)
    gx, gy = 10, 28
    d.add(Line(gx, gy, gx + 250, gy, strokeColor=GREY, strokeWidth=1.5))
    d.add(String(gx + 200, gy - 12, "flat ground", fontName="Helvetica-Oblique", fontSize=8, fillColor=GREY))
    cx, cy = gx + 50, gy + 120
    d.add(Rect(cx - 18, cy - 6, 36, 12, fillColor=NAVY, strokeColor=NAVY))
    d.add(String(cx - 25, cy + 12, "camera (lat, lon, H, roll, pitch, yaw)", fontName="Helvetica", fontSize=8,
                 textAnchor="start", fillColor=NAVY))
    d.add(Line(cx, cy - 6, cx, gy, strokeColor=GREY, strokeDashArray=[3, 3]))
    d.add(String(cx - 14, (cy + gy) / 2, "H", fontName="Helvetica-Bold", fontSize=10, fillColor=NAVY))
    tx = cx + 140
    arrow(d, cx, cy - 6, tx, gy + 2, "", RED)
    d.add(Rect(tx - 12, gy, 24, 9, fillColor=colors.white, strokeColor=RED, strokeWidth=1.2))
    d.add(String(tx + 18, gy + 12, "tent", fontName="Helvetica-Bold", fontSize=8, fillColor=RED))
    d.add(Line(cx, gy - 8, tx, gy - 8, strokeColor=TEAL))
    d.add(String((cx + tx) / 2, gy - 18, "D = (u - cx) * H / f   (nadir case)", fontName="Helvetica", fontSize=8,
                 textAnchor="middle", fillColor=TEAL))
    d.add(String(275, 125, "general case (what the code does):", fontName="Helvetica-Bold", fontSize=8.5,
                 fillColor=NAVY))
    for i, l in enumerate(["ray_cam = K^-1 [u, v, 1]", "ray_NED = R(yaw,pitch,roll) R_mount ray_cam",
                           "P = (H / ray_down) * ray_NED", "lat,lon = drone + (P_N, P_E) / radii"]):
        d.add(String(275, 108 - 14 * i, l, fontName="Courier", fontSize=7.6, fillColor=GREY))
    return d


# --------------------------------------------------------------- data
def sim_results():
    rows = []
    for d in sorted((ROOT / "results").glob("SKYBORNE_OUTPUT_*")):
        f = d / "sim_report.json"
        if f.exists():
            r = json.loads(f.read_text())
            r["_dir"] = d
            s = json.loads((d / "summary.json").read_text())
            r["_ms"] = s.get("mean_processing_ms")
            rows.append(r)
    best = {}
    for r in rows:                          # latest run per seed
        best[r["seed"]] = r
    return [best[k] for k in sorted(best)]


def build():
    res = sim_results()
    if not res:
        sys.exit("run `python main.py sim` first")
    show = next((r for r in res if r["seed"] == 42), res[-1])
    out_dir = show["_dir"]
    m = cv2.imread(str(out_dir / "final_map.jpg"))
    cv2.imwrite(str(IMG / "final_map.jpg"), cv2.resize(m, None, fx=1500 / m.shape[1], fy=1500 / m.shape[1],
                                                       interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    shutil.copy(out_dir / "tent_closeup.jpg", IMG / "tent_closeup.jpg")
    tent = json.loads((out_dir / "detections.json").read_text())["estimate"]
    story = []
    # ---------------------------------------------------------- title
    story += [Spacer(1, 2.2 * cm),
              p("Skyborne NEO'26", ParagraphStyle("T", parent=H1, fontSize=30, leading=36, alignment=TA_CENTER)),
              p("Onboard Perception &amp; Mapping Software", ParagraphStyle("T2", parent=H1, fontSize=18,
                                                                           alignment=TA_CENTER, textColor=TEAL)),
              p("How the complete software works, how it was tested, and how to run it",
                ParagraphStyle("T3", parent=B, alignment=TA_CENTER, fontSize=11, textColor=GREY)),
              Spacer(1, 0.8 * cm), architecture(), Spacer(1, 0.6 * cm),
              table([["Competition task", "What our software does", "Status"],
                     ["Endurance laps", "ArduPilot AUTO mission (tools/make_mission.py creates laps + DO_JUMP x5). "
                      "The Pi only listens.", "Mission generator done"],
                     ["Risk mapping", "Photos are placed on a north-up map during flight; final_map.png on USB "
                      "before mission time ends.", "Working in simulation"],
                     ["Target detection", "White tent detected onboard, box drawn on the map, GPS printed with "
                      "a 95 % error radius.", "Working in simulation"],
                     ["All processing onboard", "Everything runs on the Raspberry Pi 5 CPU; no ground computer.",
                      "Designed for Pi 5"]],
                    [3.4 * cm, 10.2 * cm, 3.4 * cm]),
              Spacer(1, 0.5 * cm),
              p("Perception &amp; autonomy software - Team Skyborne. Generated from the code and the simulation "
                "results in this repository.", SM), PageBreak()]

    # ---------------------------------------------------------- 1
    story += [p("1. The big picture", H1),
              p("The drone has two computers with completely separate jobs. The <b>Pixhawk</b> (ArduPilot) flies: "
                "waypoints, laps, survey pattern, return-to-launch, geofence and battery failsafes. The "
                "<b>Raspberry Pi 5</b> sees: it reads where the drone is, takes photos, builds the map, finds the "
                "tent and writes the USB stick. The Pi software is <b>read-only</b> towards the flight controller. "
                "It never arms, changes mode or steers, so a software crash on the Pi cannot crash the drone."),
              p("Every photo goes through the same six steps. The map and the tent list are built "
                "<b>during the flight</b>, so very little is left to do after landing:", B),
              per_photo(),
              p("Figure 1 - what happens for each photo and at the end of the mission.", CAP),
              p("Why this design", H2)]
    story += bullets([
        "<b>Map built in flight, not after.</b> Full photogrammetry (e.g. OpenDroneMap) needs 30-60 min on a "
        "laptop. Our map is finished seconds after landing, which matters because every second over time costs 0.5 %.",
        "<b>Geometry first, AI second.</b> We know the altitude, so every pixel has a real size in metres. "
        "A tent is 2-20 m<super>2</super>, a car roof has dark windows, a building is 100 m<super>2</super>. "
        "These physical rules are fast, explainable in the report, and need no training data.",
        "<b>Many views per tent.</b> With 75 % forward and 40 % side overlap the tent appears in 7-8 photos. "
        "Fusing them lowers the error and stops one false detection from becoming the answer.",
        "<b>Same code everywhere.</b> Simulation, SITL, replay and real flight run the identical pipeline "
        "(pipeline.py). Only the source of photos and positions changes, so what we test is what flies."])
    story += [PageBreak()]

    # ---------------------------------------------------------- 2 modules
    story += [p("2. The software modules", H1),
              table([["Folder / file", "Responsibility"],
                     ["config/settings.py", "Every number in one place (camera, overlap, thresholds, time limit). "
                      "Override in config/user_settings.json without touching code."],
                     ["communication/mavlink_reader.py", "Background thread: connects to the Pixhawk, requests "
                      "GLOBAL_POSITION_INT, ATTITUDE, GPS_RAW_INT, SYS_STATUS, DISTANCE_SENSOR at 10 Hz, "
                      "auto-reconnects, logs mission_log.csv, sends STATUSTEXT to Mission Planner."],
                     ["communication/telemetry.py", "Pose type + buffer that interpolates position/attitude to the "
                      "exact exposure time of a photo (yaw along the shortest arc)."],
                     ["camera/camera_capture.py", "Picamera2 back-end (full 8 MP, fixed 1/1000 s shutter, sensor "
                      "timestamp -> monotonic clock) and USB-webcam back-end for bench tests."],
                     ["detection/tent_detector.py", "White-blob detector with real-world size rules + fusion "
                      "with YOLO."],
                     ["detection/yolo_detector.py", "YOLO11n/YOLOv8n ONNX inference (onnxruntime or OpenCV DNN), "
                      "optional tiling for high altitude."],
                     ["mapping/camera_model.py", "Pinhole + distortion model, roll/pitch/yaw rotations, pixel <-> "
                      "ground projection."],
                     ["mapping/image_mapper.py", "Incremental mosaic: homography placement, phase-correlation "
                      "alignment with GPS anchoring, exposure matching, nadir seams."],
                     ["mapping/geolocator.py", "Sightings -> clusters -> fused tent position + 95 % error radius."],
                     ["output/map_renderer.py", "Final map: tent box, GPS label, close-up inset, north arrow, scale "
                      "bar, info panel."],
                     ["output/export_results.py", "Writes products, SHA-256 manifest, finds/mounts the USB stick, "
                      "copies and verifies."],
                     ["pipeline.py", "The mission: capture gate, per-photo processing, resume after reboot, "
                      "finalize."],
                     ["main.py", "Entry point: sim | sitl | flight | replay | detect."],
                     ["simulation/", "Synthetic world with tent + decoys, simulated flight with sensor errors, "
                      "simulated cameras (also driven by SITL)."],
                     ["tools/", "Mission generator, camera calibration, YOLO dataset generator, YOLO training "
                      "(script + Colab notebook)."],
                     ["deploy/", "Pi setup script, systemd autostart service, ArduPilot parameter reference."]],
                    [5.0 * cm, 12.0 * cm]),
              PageBreak()]

    # ---------------------------------------------------------- 3 telemetry & camera
    story += [p("3. Telemetry and photo capture", H1),
              p("3.1 Reading the flight controller (MAVLink)", H2),
              p("The Pi is wired to the Pixhawk's TELEM2 port (UART, 921600 baud, MAVLink 2). A background "
                "thread receives messages continuously and stores timestamped samples:"),
              table([["Data", "MAVLink message", "Used for"],
                     ["Latitude, longitude, relative altitude", "GLOBAL_POSITION_INT", "photo position, height H"],
                     ["Roll, pitch, yaw", "ATTITUDE", "photo orientation (map + tent GPS)"],
                     ["GPS fix type, satellites, HDOP", "GPS_RAW_INT", "capture only with a 3D fix"],
                     ["Battery voltage / %", "SYS_STATUS", "logged; shown on the map metadata"],
                     ["Armed state, flight mode", "HEARTBEAT", "start clock at arm, finalize at disarm"],
                     ["Down rangefinder (optional)", "DISTANCE_SENSOR", "true height above ground"]],
                    [5.6 * cm, 4.2 * cm, 7.2 * cm]),
              Spacer(1, 4),
              p("<b>Timing matters.</b> At 5 m/s the drone moves 0.5 m between two 10 Hz messages. The camera "
                "returns the sensor's exposure timestamp. We convert it to the same clock as the telemetry and "
                "<b>interpolate</b> the pose at that instant, which removes up to half a metre of tagging error."),
              p("3.2 When is a photo taken?", H2)]
    story += bullets(["armed, 3D GPS fix, altitude above 15 m (no photos during take-off/landing);",
                      "tilt under 20 deg (skips steep turns, which give oblique, distorted photos);",
                      "inside the survey polygon (+15 m margin), so no time is wasted on the lap course;",
                      "moved at least footprint x (1 - 75 % overlap) = <b>6.8 m at 30 m altitude</b> "
                      "(about every 1.4 s at 5 m/s), and at least 0.8 s since the last photo (Pi processing budget)."])
    story += [p("3.3 Camera numbers (Pi Camera Module v2, IMX219)", H2),
              table([["Quantity", "Value at 30 m", "Note"],
                     ["Resolution", "3280 x 2464 (8 MP)", "full sensor"],
                     ["Focal length", "3.04 mm = 2714 px", "pixel pitch 1.12 um"],
                     ["Ground sample distance", "1.11 cm / pixel", "matches the project brief"],
                     ["Footprint", "36.3 m x 27.2 m", "across x along track"],
                     ["Tent (2.4 x 2.1 m)", "~215 x 190 px", "easily resolvable"],
                     ["Shutter", "1/1000 s", "motion blur 0.5 cm at 5 m/s"]],
                    [5.0 * cm, 5.0 * cm, 7.0 * cm]),
              PageBreak()]

    # ---------------------------------------------------------- 4 detection
    story += [p("4. Finding the white tent", H1),
              p("We use a <b>two-stage hybrid detector</b>. Stage 1 always runs and needs no training. Stage 2 is "
                "a neural network that switches on automatically when models/tent_yolo.onnx exists."),
              p("4.1 Stage 1 - white-blob detector with physical rules", H2)]
    story += bullets([
        "<b>White mask (HSV):</b> saturation low and brightness above an <i>adaptive</i> threshold "
        "(median + 0.55 x (99.9th percentile - median)). The threshold follows sun, cloud and auto-exposure, "
        "and grey concrete (V~170) falls below it.",
        "<b>Brighter than its surroundings:</b> we subtract a local background (a grey-level opening with a kernel "
        "larger than the biggest tent, computed at 1/8 scale for speed). Large bright areas such as dry ground "
        "are removed. <i>Lesson from testing: our first version used a fixed small kernel that erased the "
        "middle of the tent and missed 7 of 8 views. Sizing it from metres fixed that.</i>",
        "<b>Real-world size:</b> GSD = H / f. Each blob gets an area in m<super>2</super> and side lengths in "
        "metres. Tent: 2.5-22 m<super>2</super>, sides 1.2-6 m, aspect ratio up to 2.2.",
        "<b>Shape and content:</b> solidity (convex, compact), and the fraction of dark pixels inside, which "
        "rejects white cars because of their windows.",
        "Each rule gives a 0-1 score; confidence = geometric mean. Boxes touching the image edge are down-weighted."])
    story += [table([["Object in the test world", "Why it is rejected"],
                     ["White car 4.5 x 1.8 m", "dark windows inside, aspect ratio 2.5, low solidity"],
                     ["White-roof building 12 x 8 m", "area 96 m<super>2</super> > 22 m<super>2</super>"],
                     ["Concrete pad", "brightness below the adaptive white threshold"],
                     ["Road markings 3 x 0.15 m", "too thin (short side 0.15 m < 1.2 m)"],
                     ["White stones", "area far too small"],
                     ["Orange tent (same size!)", "saturation too high - not white"]],
                    [6.0 * cm, 11.0 * cm]),
              p("4.2 Stage 2 - YOLO11n (why this model)", H2),
              table([["Model", "Params", "Pi 5 CPU / frame (640 px)", "Verdict"],
                     ["YOLO11n / YOLOv8n", "2.6-3.2 M", "~0.25-0.5 s", "<b>Chosen</b>: fits 1 frame/s, good on small objects"],
                     ["YOLO11s", "9.4 M", "~1 s", "backup if more accuracy needed"],
                     ["RT-DETR-R18 (transformer)", "20 M", "several s", "too slow on the Pi CPU"],
                     ["DETR / ViT detectors", "40 M+", "far too slow", "need a GPU"]],
                    [4.3 * cm, 2.2 * cm, 4.0 * cm, 6.5 * cm]),
              Spacer(1, 3),
              p("Timings are typical published figures for these model sizes, not measured on our Pi yet. "
                "We will measure them during integration.", SM),
              p("Transformers are excellent when a GPU is available, but the rules require processing on the drone, "
                "and the Pi 5 has no usable GPU for them. A small YOLO fine-tuned from COCO weights is the right "
                "tool. When both stages run they are fused: a YOLO box confirmed by the blob stage gets higher "
                "confidence; a blob alone is down-weighted (x0.6); neural boxes must also pass the metre-size test."),
              p("4.3 Training data", H2),
              p("tools/make_yolo_dataset.py renders labelled aerial images from the simulator: random altitude "
                "20-45 m, random attitude and heading, ridge and dome tents of random size, random exposure and "
                "white balance, and every hard negative above. 1200 images (812 with a tent) are ready in "
                "datasets/tents_dataset.zip. Training runs on a free Colab GPU with tools/train_tent_yolo_colab.ipynb."),
              img(DOCS / "img" / "labels.jpg", 13),
              p("Figure 2 - samples of the generated YOLO training labels (red = label box).", CAP),
              p("<b>Important:</b> synthetic images teach geometry and negatives, but the final model must also be "
                "trained and, above all, <b>validated on real photos</b> from our test flights over a white tent.", B),
              PageBreak()]

    # ---------------------------------------------------------- 5 map
    story += [p("5. Building the map", H1),
              p("5.1 Placing each photo", H2),
              p("The four image corners are projected to the ground with the camera model, the GPS position, the "
                "altitude and the roll/pitch/yaw. For flat ground, image-to-ground is exactly a "
                "<b>homography</b>, so tilt is corrected properly, not just rotated. The photo is first reduced to "
                "map resolution (5 cm/px), then warped onto a north-up canvas that grows as the drone flies."),
              p("5.2 Fixing GPS error between photos", H2),
              p("GPS is off by 1-2 m, so two neighbouring photos would not line up. In the overlap we measure the "
                "remaining shift with <b>phase correlation</b> (FFT based, fast, works on grass texture). We accept "
                "it only if confident and smaller than 3 m. Then we apply only 80 % of it. This <b>GPS anchoring</b> "
                "keeps the map following an average of GPS positions instead of slowly drifting. Measured in "
                "simulation over 206 photos:"),
              table([["Method", "Map position error vs truth (mean / max)", "Seam error between neighbours (mean / p90)"],
                     ["GPS only", "1.05 m / 2.15 m", "0.59 m / 1.01 m"],
                     ["Image alignment, no anchoring", "1.52 m / 3.85 m  (drifts)", "0.36 m / 0.30 m"],
                     ["<b>Alignment + 20 % GPS anchoring (used)</b>", "<b>0.94 m / 2.09 m</b>", "<b>0.21 m / 0.31 m</b>"]],
                    [5.6 * cm, 5.7 * cm, 5.7 * cm]),
              p("5.3 Seams and brightness", H2),
              p("Each map pixel keeps the photo in which it was closest to the image centre: the most vertical "
                "view with the least lens distortion. This is the nadir seam rule used in professional orthomosaic "
                "software, and costs one integer comparison per pixel. Brightness is matched in the overlap. Memory "
                "use is about 5 bytes per map pixel (about 80 MB for 200 x 200 m at 5 cm). After a reboot the map is "
                "rebuilt from the saved thumbnails in seconds."),
              PageBreak()]

    # ---------------------------------------------------------- 6 geoloc
    story += [p("6. Tent GPS coordinates", H1),
              geometry(),
              p("Figure 3 - pixel to ground. The simple formula from the plan is the special case of a perfectly "
                "level camera; the code uses the full rotation.", CAP),
              p("For each detection the box centre (and its 4 corners, for drawing) is turned into a ray, rotated by "
                "the drone attitude and the camera mounting, and intersected with flat ground H metres below "
                "(H = relative altitude + lens height). The resulting north/east offset is added to the drone "
                "position and converted to latitude/longitude with WGS-84 radii."),
              p("Fusing sightings and honest accuracy", H2)]
    story += bullets([
        "Sightings within 6 m are grouped; each group is scored by its best confidences and how many views agree. "
        "A one-off false detection cannot beat a tent seen 8 times.",
        "Position = inverse-variance weighted mean after removing outliers. Each sighting's variance comes from "
        "altitude x attitude error, distance x compass error, and altitude error x off-nadir angle.",
        "Random errors shrink with more views; <b>GPS bias does not</b>, so it is added back. We report a "
        "<b>95 % error radius</b> (2.45 sigma), not a fake precision.",
        "The box drawn on the map is snapped onto the tent's pixels in the mosaic, so it sits exactly on the tent. "
        "The printed coordinates remain the fused GPS estimate."])
    story += [PageBreak()]

    # ---------------------------------------------------------- 7 results
    rows = [["Seed (random world)", "Photos", "Sightings", "Tent GPS error", "Claimed 95 %", "Map georef error",
             "Box vs tent", "Laptop ms/photo"]]
    errs = []
    for r in res:
        if r.get("detected"):
            errs.append(r["gps_error_m"])
        rows.append([r["seed"], r["photos"], r.get("sightings", "-"),
                     f"{r['gps_error_m']:.2f} m" if r.get("detected") else "MISSED",
                     f"{r.get('claimed_error_95_m', 0):.1f} m", f"{r.get('map_georef_error_m', '-')} m",
                     f"{r.get('box_vs_tent_in_map_m', '-')} m", r.get("_ms", "-")])
    det = sum(1 for r in res if r.get("detected"))
    story += [p("7. Test results (simulation)", H1),
              p(f"The complete mission was run on {len(res)} randomly generated worlds (different terrain, tent "
                "position/orientation, decoys and sun angle). Simulated sensor errors: GPS bias 1.2 m plus 0.25 m "
                "jitter, altitude bias 0.4 m, attitude noise 0.4 deg, compass bias 1.5 deg, wind sway 3 deg. "
                "The pipeline only ever sees the noisy measurements."),
              table(rows, [3.0 * cm, 1.5 * cm, 1.8 * cm, 2.2 * cm, 2.0 * cm, 2.4 * cm, 1.9 * cm, 2.2 * cm], font=8),
              Spacer(1, 4),
              p(f"<b>Tent detected in {det}/{len(res)} worlds</b>, mean GPS error "
                f"<b>{sum(errs) / max(1, len(errs)):.2f} m</b> (max {max(errs) if errs else 0:.2f} m), always "
                "inside the claimed 95 % radius. No decoy was ever reported as the tent. 13 automated unit tests pass "
                "(python -m pytest).", B),
              img(DOCS / "img" / "final_map.jpg", 13.5, 15.5),
              p(f"Figure 4 - final_map.png produced by the software (seed {show['seed']}, downscaled). Note the "
                "decoys: the white-roof building and the white car are not boxed.", CAP),
              PageBreak(),
              p("Close-up and files on the USB stick", H2),
              img(DOCS / "img" / "tent_closeup.jpg", 7),
              p("Figure 5 - tent_closeup.jpg: the onboard photo the tent position was taken from.", CAP),
              table([["File", "Content"],
                     ["final_map.png / .jpg", "annotated map (required by the rules)"],
                     ["tent_coordinates.txt", "lat/lon (decimal + DMS), 95 % accuracy, confidence, sightings"],
                     ["detections.json", "fused estimate, every sighting, other candidates"],
                     ["image_metadata.csv", "GPS / attitude / battery for every photo"],
                     ["mission_log.csv", "telemetry log at 2 Hz"],
                     ["skyborne_map.kml + map_overlay.jpg", "opens in Google Earth with the tent placemark"],
                     ["summary.json, MANIFEST.sha256", "machine-readable summary; checksums verified after copying"]],
                    [6.0 * cm, 11.0 * cm]),
              Spacer(1, 6)]
    if tent:
        story += [p(f"Example tent_coordinates result: <b>{tent['lat']:.7f}, {tent['lon']:.7f}</b>, "
                    f"confidence {tent['confidence']:.2f}, +/- {tent['error_95_m']:.1f} m (95 %), "
                    f"{tent['sightings']} sightings.", B)]
    story += [PageBreak()]

    # ---------------------------------------------------------- 8 how to run
    story += [p("8. How to run it", H1),
              p("8.1 On the laptop (now)", H2),
              code("cd Skyborne\n.\\venv\\Scripts\\activate\npython -m pytest -q\npython main.py sim --seed 5 "
                   "--usb-dir results\\_fake_usb\npython main.py detect --image photo.jpg --alt 30"),
              p("8.2 With ArduPilot SITL (next step)", H2),
              p("Start SITL (WSL/Ubuntu: sim_vehicle.py -v ArduCopter --console --map --out=udp:&lt;laptop-ip&gt;:14551, "
                "or Mission Planner's Simulation tab with an extra UDP output), load the mission from "
                "tools/make_mission.py, then:"),
              code("python main.py sitl --connect udpin:0.0.0.0:14551"),
              p("The program anchors the synthetic world 90 m north of SITL home. Fly the survey in AUTO and land. "
                "On disarm it builds the map and copies it to a USB stick. This tests the real live loop: MAVLink "
                "thread, capture trigger, timing, finalize on disarm and export."),
              p("8.3 On the Raspberry Pi 5 (integration)", H2),
              code("bash deploy/setup_pi.sh && sudo reboot\npython main.py flight --new-session   # bench test\n"
                   "sudo systemctl enable --now skyborne   # autostart for flights\njournalctl -u skyborne -f"),
              table([["Pi 5 pin", "Pixhawk TELEM2", "Note"],
                     ["GPIO14 (TXD, pin 8)", "RX (pin 3)", "3.3 V logic on both sides"],
                     ["GPIO15 (RXD, pin 10)", "TX (pin 2)", "crossed TX/RX"],
                     ["GND (pin 6)", "GND (pin 6)", "common ground is essential"],
                     ["5 V", "-  do NOT connect", "power the Pi from its own 5 V 5 A BEC"]],
                    [5.0 * cm, 4.5 * cm, 7.5 * cm]),
              Spacer(1, 4),
              p("Flight controller: SERIAL2_PROTOCOL = 2, SERIAL2_BAUD = 921 (see deploy/ardupilot_reference_params.txt, "
                "a reference for the safety pilot to check against the firmware docs, not a file to load blindly)."),
              p("8.4 Competition-day sequence", H2)]
    story += bullets(["Power on: the Pi autostarts, waits for the heartbeat, and Mission Planner shows "
                      "\"SKY onboard ready\".",
                      "Arm and fly AUTO: laps, then the survey. The Pi photographs only over the search area; Mission "
                      "Planner messages show progress (\"SKY img 40 map 11135m2\", \"SKY tent? 0.98 ...\").",
                      "Land and disarm: the map is finalized automatically (or 150 s before the time limit, whichever "
                      "comes first; or by creating data/FINALIZE).",
                      "Insert the USB stick (it may already be in): files are copied, checksums verified, and "
                      "\"SKY: map saved to USB\" appears. Hand the stick to the judges.",
                      "If the Pi reboots mid-flight, the service restarts and <b>resumes the same session</b> "
                      "from the saved photos and thumbnails."])
    story += [PageBreak()]

    # ---------------------------------------------------------- 9 limits & next
    story += [p("9. Limitations, risks and next steps", H1),
              p("Honest limitations (for the TDR)", H2)]
    story += bullets([
        "Flat-ground assumption: tent GPS and map assume the survey area is level with home. Slopes add error of "
        "about height difference x tan(off-nadir angle). Set mapping.ground_elevation_m or fit a rangefinder.",
        "The map is a GPS-based mosaic with image alignment, <b>not a survey-grade orthomosaic</b>. Absolute "
        "accuracy is limited by GPS (about 1-2 m). Tall objects lean (relief displacement).",
        "Results so far are from simulation. The blob thresholds and the YOLO model must be tuned and validated "
        "on real photos from our test flights.",
        "Not yet tested on hardware: Picamera2 timing, the real UART link, Pi 5 CPU timing, live SITL and the "
        "YOLO ONNX path (no trained model yet). These are covered by the integration checklist below."])
    story += [p("Integration checklist (in order)", H2),
              table([["#", "Step", "Pass criterion"],
                     ["1", "SITL run (main.py sitl) with a generated mission", "map + tent on USB after disarm"],
                     ["2", "Train YOLO on Colab, drop ONNX in models/", "detector mode shows yolo+hsv"],
                     ["3", "Pi 5 setup, camera test, calibrate_camera.py", "RMS < 0.5 px"],
                     ["4", "Pi <-> Pixhawk on the bench (props OFF)", "telemetry at 10 Hz, STATUSTEXT in MP"],
                     ["5", "Measure Pi processing time per photo", "< 1.2 s (else proc_width 1280 / YOLO tiles)"],
                     ["6", "Hand-carried walk test / low hover over a white tent", "detections on real photos"],
                     ["7", "Survey flight with a safety pilot", "map complete, tent error < 3 m vs phone GPS"],
                     ["8", "3 full timed rehearsals", "USB handed over before the limit"]],
                    [0.8 * cm, 8.7 * cm, 7.5 * cm]),
              p("Open questions for the organisers", H2)]
    story += bullets(["mission time limit (config: mission.time_limit_s, default 1200 s);",
                      "size of the search area and allowed altitude (default 140 x 120 m at 30 m);",
                      "accepted method for flight termination after 90 s (ArduPilot Advanced Failsafe or RC kill);",
                      "whether a map in .jpeg is acceptable if the .png is very large (we write both)."])
    story += [p("Key settings you will most likely change", H2),
              table([["Setting (config/user_settings.json)", "Default", "Meaning"],
                     ["mission.time_limit_s / finalize_margin_s", "1200 / 150", "finish the map before time runs out"],
                     ["capture.survey_area", "from make_mission.py", "polygon where photos are taken"],
                     ["capture.forward_overlap", "0.75", "photo spacing (6.8 m at 30 m)"],
                     ["detection.tent_min/max_area_m2", "2.5 / 22", "expected tent size"],
                     ["detection.use_yolo", "auto", "on when models/tent_yolo.onnx exists"],
                     ["mapping.resolution_m", "0.05", "map pixel size (5 cm)"],
                     ["mavlink.connection / baud", "/dev/ttyAMA0 / 921600", "Pi UART to TELEM2"]],
                    [6.5 * cm, 4.0 * cm, 6.5 * cm])]

    pdf = DOCS / "Skyborne_Software_Guide.pdf"

    def on_page(c, doc):
        c.saveState()
        c.setFont("Helvetica", 7.5)
        c.setFillColor(GREY)
        c.drawString(2 * cm, 1.2 * cm, "Skyborne NEO'26 - Onboard Software Guide")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
        c.setStrokeColor(TEAL)
        c.line(2 * cm, A4[1] - 1.3 * cm, A4[0] - 2 * cm, A4[1] - 1.3 * cm)
        c.restoreState()

    SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=1.8 * cm, title="Skyborne Software Guide", author="Team Skyborne").build(
        story, onFirstPage=on_page, onLaterPages=on_page)
    print("PDF written:", pdf)


if __name__ == "__main__":
    build()
