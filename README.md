# Skyborne NEO'26 — onboard perception & mapping software

Runs on the Raspberry Pi 5. It reads telemetry from the Pixhawk (read-only MAVLink), takes photos pointing straight down, builds a geo-referenced map during the flight, finds the white tent, estimates its GPS position, and writes the results to a USB stick.
**ArduPilot flies the drone and handles every failsafe. This software never sends flight commands.**

Full explanation: `docs/Skyborne_Software_Guide.pdf`.

## Quick start (laptop, no hardware)
```powershell
python -m venv venv; .\venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                       # 13 unit tests
python main.py sim --usb-dir results\_fake_usb   # full simulated mission, scored against ground truth
```
Output appears in `results/SKYBORNE_OUTPUT_<date>_<time>/`: `final_map.png`, `tent_coordinates.txt`, `detections.json`, `image_metadata.csv`, `mission_log.csv`, `tent_closeup.jpg`, `skyborne_map.kml`, and `MANIFEST.sha256`.

## Modes
| Command | What it does |
|---|---|
| `python main.py sim [--seed N]` | Synthetic world and a simulated survey flight with GPS, attitude and compass errors |
| `python main.py sitl [--connect udpin:0.0.0.0:14551]` | ArduPilot SITL telemetry with a synthetic camera; the full live loop |
| `python main.py flight` | On the Pi: Pixhawk on `/dev/ttyAMA0` (TELEM2) and Pi Camera v2 |
| `python main.py replay --session data/sessions/<dir>` | Re-process a recorded flight on the laptop |
| `python main.py detect --image a.jpg --alt 30` | Run the detector on one photo |
| `python -m communication.mavlink_reader --connect ...` | Print live telemetry |

## Layout
```
config/settings.py          all tunable numbers (override in config/user_settings.json)
communication/              MAVLink reader thread + pose interpolation buffer
camera/                     Picamera2 / USB camera back-ends, metadata CSV
detection/                  white-blob detector, YOLO (ONNX) detector, fusion
mapping/                    camera model, GPS<->metres, mosaic, tent geolocation
output/                     map rendering, USB export with SHA-256 verification
simulation/                 synthetic world, simulated flight, simulated cameras
pipeline.py                 the onboard pipeline (same code in every mode)
tools/                      mission generator, camera calibration, YOLO dataset + training
deploy/                     Pi setup script, systemd service, ArduPilot parameter reference
```

## Raspberry Pi 5
```bash
bash deploy/setup_pi.sh && sudo reboot
python main.py flight --new-session        # first manual test
sudo systemctl enable --now skyborne       # autostart; resumes the session after a reboot
```
Wiring: Pi GPIO14 (TX) → TELEM2 RX, GPIO15 (RX) ← TELEM2 TX, and GND ↔ GND (both sides use 3.3 V logic). Power the Pi from a separate 5 V 5 A BEC.

## YOLO model (optional, recommended)
1. `python tools/make_yolo_dataset.py --n 1500` generates synthetic labelled images (`datasets/tents_dataset.zip` is already built). Add real photos.
2. Train on Colab with `tools/train_tent_yolo_colab.ipynb`. It produces `tent_yolo.onnx`.
3. Copy it to `models/tent_yolo.onnx`. The detector switches to `yolo+hsv` automatically.
