#!/usr/bin/env python3
"""Pre-game diagnostics for camera, microphone, and system health.

This script verifies the availability of camera and microphone devices,
checks that a frame can be captured from the camera, measures the audio RMS
level, and reports basic system metrics.  Each section prints a pass/fail
indicator to help operators quickly confirm the rig is ready for streaming.
"""

from __future__ import annotations

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

import glob
import os
import re
import shutil
import subprocess
from typing import Tuple

try:  # optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 may be unavailable
    cv2 = None  # type: ignore

try:  # optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil may be unavailable
    psutil = None  # type: ignore


PASS = "✅ PASSED"
FAIL = "❌ FAILED"


def check_camera(device: str = "/dev/video0", index: int = 0) -> Tuple[bool, str]:
    """Verify that a camera exists and can capture one frame."""

    if not os.path.exists(device):
        return False, f"{device} missing"
    if cv2 is None:
        return False, "cv2 not installed"
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return False, "cannot open camera"
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return False, "read failed"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return True, f"{width}x{height}"


def check_microphone(device: str = "hw:1,0", seconds: int = 3) -> Tuple[bool, str]:
    """Measure audio RMS level using ffmpeg."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg missing"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "alsa",
        "-i",
        device,
        "-t",
        str(seconds),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 5)
    except Exception:
        return False, "record failed"
    if proc.returncode != 0:
        return False, "device not available"
    match = re.search(r"mean_volume:\s*([-0-9.]+) dB", proc.stderr)
    if not match:
        return False, "no audio"
    mean_db = float(match.group(1))
    rms = 10 ** (mean_db / 20)
    return True, f"RMS {rms:.3f}"


def system_usage() -> Tuple[bool, str]:
    """Return CPU and GPU usage percentages."""

    cpu = psutil.cpu_percent(interval=1) if psutil else None
    tegra = shutil.which("tegrastats")
    gpu = None
    if tegra:
        try:
            out = subprocess.check_output(
                [tegra, "--interval", "1000", "--count", "1"], text=True
            )
            m = re.search(r"GR3D_FREQ\s+(\d+)%", out)
            if m:
                gpu = int(m.group(1))
        except Exception:
            pass
    ok = cpu is not None and gpu is not None
    cpu_txt = f"CPU {cpu:.1f}%" if cpu is not None else "CPU N/A"
    gpu_txt = f"GPU {gpu}%" if gpu is not None else "GPU N/A"
    return ok, f"{cpu_txt} | {gpu_txt}"


def board_temp() -> Tuple[bool, str]:
    """Read Jetson board temperature in Celsius."""

    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(path, "r", encoding="utf8") as fh:
                val = int(fh.read().strip()) / 1000.0
                return True, f"{val:.1f}C"
        except Exception:
            continue
    return False, "temp not found"


def stream_key_present() -> Tuple[bool, str]:
    """Check if a stream key is available via environment variables."""

    key = os.getenv("YOUTUBE_STREAM_KEY") or os.getenv("STREAM_KEY")
    if key and "YOUR_STREAM_KEY" not in key:
        return True, "key set"
    return False, "missing"


def print_result(name: str, result: Tuple[bool, str]) -> None:
    ok, info = result
    status = PASS if ok else FAIL
    logging.info(f"{name}: {status} {info}")
def main() -> None:  # pragma: no cover - CLI helper
    print_result("Camera /dev/video0", check_camera())
    print_result("Microphone hw:1,0", check_microphone())
    print_result("System", system_usage())
    print_result("Board Temp", board_temp())
    print_result("Stream Key", stream_key_present())


if __name__ == "__main__":
    main()
