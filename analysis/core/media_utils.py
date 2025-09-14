"""Media helpers built atop FFmpeg.

Provides tiny wrappers around ``ffprobe`` and ``ffmpeg`` for
probing and cutting clips.  These helpers favour stream
copying on keyframes and fall back to re-encoding when
necessary.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def ffprobe_json(path: str | Path) -> Dict[str, Any]:
    """Return ``ffprobe`` info for ``path`` as a dictionary."""
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def ffmpeg_cut(
    src: str | Path,
    start: float,
    end: float,
    dst: str | Path,
    *,
    prefer_copy: bool = True,
) -> None:
    """Cut ``src`` between ``start`` and ``end`` seconds to ``dst``.

    ``prefer_copy`` attempts stream copying which is fast and
    lossless but requires keyframe aligned cuts.  When FFmpeg
    fails, the function retries with a tiny re-encode.
    """
    dst = str(dst)
    cmd: List[str] = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        str(src),
    ]
    if prefer_copy:
        copy_cmd = cmd + ["-c", "copy", dst]
        proc = subprocess.run(copy_cmd)
        if proc.returncode == 0:
            return
    re_cmd = cmd + ["-c:v", "libx264", "-c:a", "aac", dst]
    subprocess.check_call(re_cmd)


__all__ = ["ffprobe_json", "ffmpeg_cut"]
