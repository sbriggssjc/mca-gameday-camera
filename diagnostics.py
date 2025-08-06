"""Simple system diagnostics for gameday camera deployments.

This utility reports CPU usage, approximate camera FPS, encoder
bitrate and basic thermal information.  It is intended for quick
health checks before or during a stream.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time

try:  # optional dependencies
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil may be unavailable
    psutil = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 may be unavailable
    cv2 = None  # type: ignore


def measure_fps(index: int, seconds: int = 2) -> float:
    """Return the average FPS for ``index`` over ``seconds``."""

    if cv2 is None:
        return 0.0
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return 0.0
    start = time.time()
    frames = 0
    while time.time() - start < seconds:
        ret, _ = cap.read()
        if ret:
            frames += 1
    cap.release()
    elapsed = max(0.001, time.time() - start)
    return frames / elapsed


def measure_bitrate(seconds: int = 2, bitrate: str = "4500k") -> float:
    """Return the achieved bitrate from a short ffmpeg run."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return 0.0
    cmd = [
        ffmpeg,
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=1280x720:rate=30",
        "-t",
        str(seconds),
        "-c:v",
        "libx264",
        "-b:v",
        bitrate,
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 5)
        match = re.search(r"bitrate=\s*(\d+\.?\d*)kbits/s", proc.stderr)
        if match:
            return float(match.group(1))
    except Exception:  # pragma: no cover - best effort
        pass
    return 0.0


def measure_temp() -> float | None:
    """Return the CPU temperature in Celsius if available."""

    if psutil and hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()
            for key in ("cpu-thermal", "cpu", "Package id 0"):
                if key in temps:
                    return temps[key][0].current
        except Exception:
            pass
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
        m = re.search(r"temp=([0-9.]+)", out)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def main() -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="Run system diagnostics")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to probe")
    parser.add_argument("--seconds", type=int, default=2, help="Seconds to sample")
    parser.add_argument("--bitrate", default="4500k", help="Target bitrate for test")
    args = parser.parse_args()

    cpu = psutil.cpu_percent(interval=1) if psutil else 0.0
    fps = measure_fps(args.camera, args.seconds)
    br = measure_bitrate(args.seconds, args.bitrate)
    temp = measure_temp()

    msg = f"CPU {cpu:.1f}% | FPS {fps:.1f} | Bitrate {br:.0f} kbps"
    if temp is not None:
        msg += f" | Temp {temp:.1f}C"
    print(msg)


if __name__ == "__main__":
    main()

