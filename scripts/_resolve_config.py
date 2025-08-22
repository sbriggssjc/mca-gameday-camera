#!/usr/bin/env python3
"""Resolve gameday configuration.

Reads configuration from disk and environment, validates required fields, and
emits a single compact JSON blob on stdout for consumption by the launcher.

Diagnostics and human friendly summaries are printed to stderr.
"""

import os
import sys
import json
import re
import subprocess


YT_HOSTS = {"a.rtmp.youtube.com", "a.rtmps.youtube.com", "b.rtmps.youtube.com"}


def _valid_rtmp(u: str) -> bool:
    """Strictly validate a YouTube RTMP(S) URL."""

    if not isinstance(u, str) or not u:
        return False
    if not (u.startswith("rtmp://") or u.startswith("rtmps://")):
        return False
    m = re.match(r"^(rtmps?://)([^/]+)(/live2/[^/\s<>]+)$", u.strip())
    if not m:
        return False
    host = m.group(2)
    return host in YT_HOSTS


def load_json(path: str):
    """Load JSON from disk; return empty dict on missing file.

    Any JSON parsing error is surfaced and terminates the program.
    """

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:  # pragma: no cover - defensive
        print(f"[gameday] bad JSON in {path}: {e}", file=sys.stderr)
        sys.exit(2)


cfg_path = os.environ.get("GAMEDAY_CONFIG", "config/gameday.json")
disk = load_json(cfg_path)

# Merge precedence: env > config file > defaults
resolved = {
    "rtmp_url": os.environ.get("YOUTUBE_RTMP_URL", disk.get("rtmp_url", "")),
    "video_dev": os.environ.get("VIDEO_DEV", disk.get("video_dev", "/dev/video0")),
    "pulse_source": os.environ.get("PULSE_DEV", disk.get("pulse_source", "")),
    "video_size": os.environ.get("VIDEO_SIZE", disk.get("video_size", "1280x720")),
    "fps": int(os.environ.get("FPS", disk.get("fps", 30))),
    "use_hw": os.environ.get("USE_HW", str(disk.get("use_hw", "auto"))).lower(),  # auto|yes|no
    "testsrc": os.environ.get("TESTSRC", "0"),  # "1" to enable test pattern mode
}


ok = True
if not _valid_rtmp(resolved["rtmp_url"]):
    print("[gameday] missing or invalid RTMP URL", file=sys.stderr)
    ok = False
if not resolved["pulse_source"]:
    print("[gameday] missing pulse_source (PULSE_DEV)", file=sys.stderr)
    ok = False


print(
    f"[gameday] Launch -> video={resolved['video_dev']} {resolved['video_size']}@{resolved['fps']} | "
    f"pulse={resolved['pulse_source']} | rtmp={'set' if _valid_rtmp(resolved['rtmp_url']) else 'MISSING'}",
    file=sys.stderr,
)

if not ok:
    sys.exit(2)

# Emit compact JSON ONLY on stdout
sys.stdout.write(json.dumps(resolved, separators=(",", ":")))

