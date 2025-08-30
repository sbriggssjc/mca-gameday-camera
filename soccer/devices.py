import os
import subprocess
from typing import Optional


def guess_video_dev() -> str:
    """Return a likely v4l2 video device."""
    if os.path.exists("/dev/video0"):
        return "/dev/video0"
    # fall back to first /dev/video* device
    for i in range(10):
        dev = f"/dev/video{i}"
        if os.path.exists(dev):
            return dev
    return "/dev/video0"  # best effort


def _pactl_list_sources() -> str:
    try:
        return subprocess.check_output(
            ["pactl", "list", "short", "sources"], text=True
        )
    except Exception:
        return ""


def guess_pulse_src() -> str:
    """Return the default PulseAudio source or a Rode mic if present."""
    sources = _pactl_list_sources().splitlines()
    # look for Rode VideoMic GO II
    for line in sources:
        parts = line.split("\t")
        if len(parts) >= 2 and "R__DE" in parts[1]:
            return parts[1]
    # try to find line marked as `*`?  pactl short doesn't show default; rely on 'default'
    return "default"
