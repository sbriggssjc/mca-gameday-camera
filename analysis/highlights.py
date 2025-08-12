"""Clip generation helpers and highlight builder."""

from __future__ import annotations

import os
import subprocess
import pathlib
import shutil
import tempfile
from typing import Tuple


def clip_range(start_s: float, end_s: float, padding: float) -> Tuple[float, float]:
    """Return a padded time range with no negative start."""

    start = max(0.0, start_s - padding)
    end = end_s + padding
    return start, end


def ensure_output_dirs(base: str, jersey: str) -> Tuple[str, str]:
    """Return paths for good/needs work clips creating directories as needed."""

    good = os.path.join(base, "players", jersey, "good")
    needs = os.path.join(base, "players", jersey, "needs_work")
    os.makedirs(good, exist_ok=True)
    os.makedirs(needs, exist_ok=True)
    return good, needs


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run ``cmd`` suppressing output and raising on failure."""

    return subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_highlight(clips_dir: pathlib.Path, out_dir: pathlib.Path):
    """Concatenate clips into a highlight reel with audio and loudness normalisation."""

    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sorted(clips_dir.glob("play_*.mp4"))
    if not clips:
        return None

    concat = out_dir / "concat.txt"
    with concat.open("w") as f:
        for c in clips:
            f.write(f"file '{c.resolve()}'\n")

    out_raw = out_dir / "window_highlight.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vsync",
            "vfr",
            "-af",
            "aresample=async=1:min_hard_comp=0.100:first_pts=0",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            str(out_raw),
        ]
    )

    out_norm = out_dir / "window_highlight_loudnorm.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(out_raw),
            "-filter:a",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(out_norm),
        ]
    )

    return out_norm
