"""
Read-only MAVLink link between the Raspberry Pi and the flight controller.

* Runs in a background thread, reconnects automatically if the cable/radio drops.
* Asks the autopilot for the messages we need at `stream_rate_hz`.
* Stores every position/attitude sample in a TelemetryBuffer so the camera
  code can ask "where was the drone at time t?".
* Sends a 1 Hz HEARTBEAT as an onboard computer and optional STATUSTEXT
  progress messages (shown in Mission Planner's Messages tab).

SAFETY: this module never arms, changes mode, or sends movement commands.
The flight controller stays fully in charge of flying and failsafes.

Standalone test (SITL or real Pixhawk):
    python -m communication.mavlink_reader --connect udpin:0.0.0.0:14551
    python -m communication.mavlink_reader --connect /dev/ttyAMA0 --baud 921600
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import queue
import threading
import time
from pathlib import Path

from communication.telemetry import CSV_FIELDS, TelemetryBuffer, VehicleStatus

log = logging.getLogger("mavlink")

# MAVLink message ids we request
MSG_IDS = {"SYS_STATUS": 1, "GPS_RAW_INT": 24, "ATTITUDE": 30, "GLOBAL_POSITION_INT": 33,
           "VFR_HUD": 74, "DISTANCE_SENSOR": 132}


class MavlinkTelemetry:
    def __init__(self, connection: str, baud: int = 921600, stream_rate_hz: float = 10.0,
                 log_path: str | Path | None = None, log_rate_hz: float = 2.0,
                 send_heartbeat: bool = True, send_statustext: bool = True,
                 heartbeat_timeout_s: float = 30.0, clock=time.monotonic):
        self.connection_str = connection
        self.baud = baud
        self.rate = stream_rate_hz
        self.clock = clock
        self.buffer = TelemetryBuffer()
        self._status = VehicleStatus()
        self._status_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._texts: queue.Queue = queue.Queue(maxsize=20)
        self._send_hb = send_heartbeat
        self._send_text = send_statustext
        self._hb_timeout = heartbeat_timeout_s
        self._log_path = Path(log_path) if log_path else None
        self._log_period = 1.0 / log_rate_hz if log_rate_hz > 0 else 1e9
        self._last_log = 0.0
        self._last_text = 0.0
        self.target_system = 1
        self.conn = None

    # ------------------------------------------------------------- public API
    def start(self) -> "MavlinkTelemetry":
        self._thread = threading.Thread(target=self._run, name="mavlink", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def status(self) -> VehicleStatus:
        with self._status_lock:
            s = self._status.copy()
        s.connected = s.connected and (self.clock() - s.last_heartbeat) < 5.0
        return s

    def pose_at(self, t):
        return self.buffer.pose_at(t)

    def latest(self):
        return self.buffer.latest()

    def send_status(self, text: str, severity: int = 6):
        """Queue a STATUSTEXT for the ground station (max 50 chars, rate limited)."""
        if self._send_text:
            try:
                self._texts.put_nowait((severity, text[:50]))
            except queue.Full:
                pass

    def wait_connected(self, timeout: float = 60.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.status().connected and self.latest() is not None:
                return True
            time.sleep(0.2)
        return False

    # --------------------------------------------------------------- internals
    def _run(self):
        from pymavlink import mavutil
        while not self._stop.is_set():
            try:
                log.info("Connecting to %s ...", self.connection_str)
                self.conn = mavutil.mavlink_connection(
                    self.connection_str, baud=self.baud, source_system=1,
                    source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER, autoreconnect=True)
                hb = self.conn.wait_heartbeat(timeout=self._hb_timeout)
                if hb is None:
                    raise TimeoutError("no heartbeat")
                self.target_system = self.conn.target_system
                log.info("Heartbeat from system %d", self.target_system)
                self._request_streams(mavutil)
                self._loop(mavutil)
            except Exception as exc:  # noqa: BLE001 - keep the link alive no matter what
                log.warning("MAVLink link error: %s - retrying in 2 s", exc)
                with self._status_lock:
                    self._status.connected = False
                try:
                    if self.conn:
                        self.conn.close()
                except Exception:  # noqa: BLE001
                    pass
                self._stop.wait(2.0)

    def _request_streams(self, mavutil):
        interval_us = int(1e6 / self.rate)
        for name, mid in MSG_IDS.items():
            self.conn.mav.command_long_send(
                self.target_system, 1, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mid, interval_us if name not in ("SYS_STATUS", "GPS_RAW_INT") else 500000, 0, 0, 0, 0, 0)
        # Older firmware fallback
        self.conn.mav.request_data_stream_send(self.target_system, 1,
                                               mavutil.mavlink.MAV_DATA_STREAM_ALL, int(self.rate), 1)

    def _loop(self, mavutil):
        last_hb_sent = 0.0
        last_rx = self.clock()
        while not self._stop.is_set():
            now = self.clock()
            if self._send_hb and now - last_hb_sent >= 1.0:
                self.conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                             mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                last_hb_sent = now
            self._flush_texts()
            msg = self.conn.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                if self.clock() - last_rx > 10.0:
                    raise TimeoutError("no telemetry for 10 s")
                continue
            last_rx = self.clock()
            self._handle(msg, mavutil, last_rx)
            if last_rx - self._last_log >= self._log_period:
                self._write_log(last_rx)

    def _handle(self, msg, mavutil, t):
        mt = msg.get_type()
        if mt == "GLOBAL_POSITION_INT":
            gs = math.hypot(msg.vx, msg.vy) / 100.0
            self.buffer.add_position(t, msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0, gs)
        elif mt == "ATTITUDE":
            self.buffer.add_attitude(t, msg.roll, msg.pitch, msg.yaw)
        elif mt == "GPS_RAW_INT":
            hdop = msg.eph / 100.0 if msg.eph != 65535 else 99.0
            self.buffer.set_gps_quality(msg.fix_type, msg.satellites_visible, hdop)
        elif mt == "DISTANCE_SENSOR":
            if msg.orientation == 25:          # MAV_SENSOR_ROTATION_PITCH_270 = facing down
                self.buffer.set_rangefinder(t, msg.current_distance / 100.0)
        elif mt == "SYS_STATUS":
            with self._status_lock:
                if msg.voltage_battery != 65535:
                    self._status.battery_v = msg.voltage_battery / 1000.0
                self._status.battery_pct = float(msg.battery_remaining)
        elif mt == "HEARTBEAT":
            if msg.type == mavutil.mavlink.MAV_TYPE_GCS or msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                return
            if msg.get_srcSystem() != self.target_system:
                return
            with self._status_lock:
                self._status.connected = True
                self._status.last_heartbeat = t
                self._status.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self._status.mode = mavutil.mode_string_v10(msg)

    def _flush_texts(self):
        if self._texts.empty() or self.clock() - self._last_text < 1.0:
            return
        sev, text = self._texts.get_nowait()
        self.conn.mav.statustext_send(sev, text.encode("ascii", "replace"))
        self._last_text = self.clock()

    def _write_log(self, t):
        self._last_log = t
        if not self._log_path:
            return
        pose = self.buffer.latest()
        st = self.status()
        if pose is None:
            return
        new = not self._log_path.exists()
        with self._log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS + ["armed", "mode", "battery_v", "battery_pct"])
            if new:
                w.writeheader()
            row = pose.to_row()
            row.update(armed=int(st.armed), mode=st.mode,
                       battery_v="" if math.isnan(st.battery_v) else round(st.battery_v, 2),
                       battery_pct=st.battery_pct)
            w.writerow(row)


def _cli():
    ap = argparse.ArgumentParser(description="Print live telemetry (read-only)")
    ap.add_argument("--connect", default="udpin:0.0.0.0:14551")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--log", default=None, help="CSV file to save telemetry")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    tel = MavlinkTelemetry(a.connect, a.baud, log_path=a.log).start()
    print("Waiting for autopilot heartbeat ...")
    tel.wait_connected(120)
    try:
        while True:
            p, s = tel.latest(), tel.status()
            if p:
                print(f"{s.mode:>8} armed={int(s.armed)}  lat {p.lat:.7f} lon {p.lon:.7f}  alt {p.alt:6.1f} m  "
                      f"r/p/y {math.degrees(p.roll):5.1f} {math.degrees(p.pitch):5.1f} {math.degrees(p.yaw) % 360:5.1f}  "
                      f"gs {p.groundspeed:4.1f} m/s  fix {p.gps_fix} sats {p.satellites}  batt {s.battery_v:.2f} V {s.battery_pct:.0f}%")
            else:
                print("no telemetry yet")
            time.sleep(1)
    except KeyboardInterrupt:
        tel.stop()


if __name__ == "__main__":
    _cli()
