#!/usr/bin/env bash
# One-time Raspberry Pi 5 setup for Skyborne (Raspberry Pi OS Bookworm 64-bit).
# Copy the Skyborne folder to /home/<user>/Skyborne, then:  bash deploy/setup_pi.sh
set -euo pipefail
cd "$(dirname "$0")/.."
APP_DIR="$(pwd)"

echo "== system packages"
sudo apt update
sudo apt install -y python3-venv python3-pip python3-picamera2 python3-opencv python3-numpy udisks2

echo "== UART for the Pixhawk (GPIO14 TX -> TELEM2 RX, GPIO15 RX <- TELEM2 TX, GND <-> GND)"
# disable the Linux serial console on the UART, keep the hardware enabled
sudo raspi-config nonint do_serial_cons 1 || true
sudo raspi-config nonint do_serial_hw 0 || true
CFG=/boot/firmware/config.txt
grep -q '^dtparam=uart0=on' "$CFG" || echo 'dtparam=uart0=on' | sudo tee -a "$CFG"
# camera auto-detect is on by default; for the Pi Camera v2 (IMX219) on CAM0 you can force it with:
grep -q '^camera_auto_detect=1' "$CFG" || echo 'camera_auto_detect=1' | sudo tee -a "$CFG"

echo "== Python environment (uses the OS picamera2/opencv/numpy)"
python3 -m venv --system-site-packages venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements-pi.txt

echo "== autostart service"
sed "s#__APP_DIR__#${APP_DIR}#g" deploy/skyborne.service | sudo tee /etc/systemd/system/skyborne.service >/dev/null
sudo systemctl daemon-reload
echo "Enable autostart with:   sudo systemctl enable --now skyborne"
echo "Live log:                journalctl -u skyborne -f"

echo "== quick checks"
./venv/bin/python -c "from picamera2 import Picamera2; print('picamera2 OK')"
./venv/bin/python -c "import cv2, numpy, pymavlink; print('opencv', cv2.__version__, 'OK')"
ls -l /dev/ttyAMA0 2>/dev/null || echo "NOTE: /dev/ttyAMA0 appears after a reboot"
echo "Reboot now:  sudo reboot"
