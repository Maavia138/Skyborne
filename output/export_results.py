"""
Write the final products and hand them over on a USB stick.

Products (folder SKYBORNE_OUTPUT_<date>_<time>):
    final_map.png          full-resolution annotated map  (required by the rules)
    final_map.jpg          same, smaller file
    tent_coordinates.txt   human-readable tent GPS
    detections.json        all sightings + fused estimate
    image_metadata.csv     GPS/attitude of every photo
    mission_log.csv        telemetry log
    tent_closeup.jpg       onboard photo of the tent with box
    skyborne_map.kml       open in Google Earth: map overlay + tent placemark
    summary.json           machine-readable summary
    MANIFEST.sha256        checksums (verified after copying)

USB detection: Linux /proc/mounts (/media, /mnt, /run/media) or Windows
removable drives. The copy is verified with SHA-256 before reporting success.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import time
from pathlib import Path

log = logging.getLogger("export")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(folder: Path) -> Path:
    lines = [f"{sha256(p)}  {p.name}" for p in sorted(folder.iterdir()) if p.is_file() and p.name != "MANIFEST.sha256"]
    m = folder / "MANIFEST.sha256"
    m.write_text("\n".join(lines) + "\n")
    return m


def find_usb_mounts() -> list[Path]:
    found: list[Path] = []
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/mounts").read_text().splitlines():
                dev, mnt, fstype = line.split()[:3]
                mnt = mnt.replace("\\040", " ")
                if dev.startswith("/dev/sd") and fstype in ("vfat", "exfat", "ntfs", "ntfs3", "fuseblk", "ext4") \
                        and mnt.startswith(("/media", "/mnt", "/run/media")):
                    found.append(Path(mnt))
        except OSError:
            pass
    elif sys.platform == "win32":
        import ctypes
        import string
        mask = ctypes.windll.kernel32.GetLogicalDrives()
        for i, letter in enumerate(string.ascii_uppercase):
            if mask & (1 << i):
                root = f"{letter}:\\"
                if ctypes.windll.kernel32.GetDriveTypeW(root) == 2:   # DRIVE_REMOVABLE
                    found.append(Path(root))
    found = [p for p in found if os.access(p, os.W_OK)]
    if not found and sys.platform.startswith("linux"):
        found = _try_mount_linux()
    return found


def _try_mount_linux() -> list[Path]:
    """Headless Pi OS does not auto-mount USB sticks: mount the first USB partition ourselves."""
    import glob
    import subprocess
    for dev in sorted(glob.glob("/dev/sd[a-z][0-9]")) or sorted(glob.glob("/dev/sd[a-z]")):
        try:
            r = subprocess.run(["udisksctl", "mount", "-b", dev, "--no-user-interaction"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and " at " in r.stdout:
                return [Path(r.stdout.strip().split(" at ", 1)[1].rstrip("."))]
        except (OSError, subprocess.SubprocessError):
            pass
        mnt = Path("/mnt/skyborne_usb")
        try:
            mnt.mkdir(parents=True, exist_ok=True)
            if subprocess.run(["mount", dev, str(mnt)], capture_output=True, timeout=10).returncode == 0:
                return [mnt]
        except (OSError, subprocess.SubprocessError):
            pass
    return []


def copy_to_usb(src: Path, usb_root: Path) -> tuple[bool, Path, str]:
    dst = usb_root / src.name
    try:
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for p in src.iterdir():
            if p.is_file():
                with p.open("rb") as fi, (dst / p.name).open("wb") as fo:
                    shutil.copyfileobj(fi, fo, 1 << 20)
                    fo.flush()
                    os.fsync(fo.fileno())
        if hasattr(os, "sync"):
            os.sync()
        # verify every file against the local checksum
        for line in (src / "MANIFEST.sha256").read_text().splitlines():
            digest, name = line.split("  ", 1)
            if sha256(dst / name) != digest:
                return False, dst, f"checksum mismatch: {name}"
        return True, dst, "ok"
    except OSError as exc:
        return False, dst, str(exc)


def export_to_usb(src: Path, usb_path: str | None = None, wait_s: float = 0.0, notify=None) -> dict:
    """Copy `src` to the USB stick, waiting up to wait_s for one to appear."""
    end = time.time() + wait_s
    announced = False
    while True:
        targets = [Path(usb_path)] if usb_path else find_usb_mounts()
        targets = [t for t in targets if t.exists()]
        for t in targets:
            ok, dst, msg = copy_to_usb(src, t)
            if ok:
                log.info("USB handover complete: %s", dst)
                if notify:
                    notify("SKY: map saved to USB")
                return {"ok": True, "path": str(dst)}
            log.error("USB copy to %s failed: %s", t, msg)
        if time.time() >= end:
            log.warning("No USB stick written - results remain in %s", src)
            return {"ok": False, "path": str(src), "error": "no writable USB drive"}
        if not announced:
            log.info("Waiting for USB stick (up to %.0f s) ...", wait_s)
            if notify:
                notify("SKY: insert USB stick")
            announced = True
        time.sleep(2.0)


def write_kml(path: Path, map_image_name: str, bounds: dict, estimate) -> None:
    placemark = ""
    if estimate is not None:
        placemark = f"""
  <Placemark><name>White tent (conf {estimate.confidence:.2f}, +/-{estimate.error_95_m:.1f} m)</name>
    <Point><coordinates>{estimate.lon:.8f},{estimate.lat:.8f},0</coordinates></Point></Placemark>"""
    path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Skyborne map</name>
  <GroundOverlay><name>Skyborne mosaic</name><Icon><href>{map_image_name}</href></Icon>
    <LatLonBox><north>{bounds['north']:.8f}</north><south>{bounds['south']:.8f}</south>
      <east>{bounds['east']:.8f}</east><west>{bounds['west']:.8f}</west></LatLonBox></GroundOverlay>{placemark}
</Document></kml>
""")
