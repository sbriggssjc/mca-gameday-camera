"""Utilities to stack two video streams using ffmpeg."""
from __future__ import annotations

import shutil
import subprocess
from typing import Literal


def stack(original: str, aerial: str, out_path: str, *, align: Literal["h", "v"] = "h") -> bool:
    """Stack ``original`` and ``aerial`` videos side-by-side.

    ``align`` controls whether the videos are stacked horizontally (``"h"``) or
    vertically (``"v"``).  The function returns ``True`` when stacking succeeds
    and ``False`` otherwise.  Audio is preserved from the original clip when
    possible.
    """

    if shutil.which("ffmpeg") is None:  # pragma: no cover - safeguard
        return False
    filt = "hstack" if align == "h" else "vstack"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        original,
        "-i",
        aerial,
        "-filter_complex",
        f"{filt}=inputs=2",
        "-c:v",
        "libx264",
        "-c:a",
        "copy",
        out_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode == 0
