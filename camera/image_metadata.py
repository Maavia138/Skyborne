"""
Per-image metadata (GPS + attitude at exposure time) stored as CSV.

Every photo is written next to one CSV row, so a flight can be re-processed
later on the laptop (`python main.py replay --session ...`) and the onboard
map can be rebuilt after an unexpected reboot.
"""
from __future__ import annotations

import csv
import threading
from pathlib import Path

from communication.telemetry import CSV_FIELDS, Pose

FIELDS = ["image", "frame_id"] + CSV_FIELDS + ["battery_pct", "mode"]


class MetadataWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def write(self, image_name: str, frame_id: int, pose: Pose, battery_pct=-1, mode=""):
        row = {"image": image_name, "frame_id": frame_id, **pose.to_row(),
               "battery_pct": battery_pct, "mode": mode}
        with self._lock, self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


def read_metadata(path: str | Path) -> list[tuple[str, int, Pose]]:
    out = []
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            out.append((r["image"], int(r.get("frame_id") or len(out)), Pose.from_row(r)))
    return out
