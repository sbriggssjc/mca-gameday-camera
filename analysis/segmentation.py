from __future__ import annotations

import os
import subprocess
import tempfile
from typing import List, Dict

import numpy as np

try:  # pragma: no cover
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

from .features import audio_rms_peaks, motion_activity_times, find_play_windows


def _video_duration(path: str) -> float:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ]
        )
        return float(out.strip() or 0.0)
    except Exception:
        return 0.0


def _ensure_wav(video_path: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        tmp.name,
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tmp.name


def segment_video(
    video_path: str,
    min_play_length: float = 3.0,
    max_play_length: float = 12.0,
    min_play_gap: float = 1.5,
    preroll: float = 0.75,
    postroll: float = 0.75,
) -> List[Dict]:
    """
    Returns a list of segments: [{"t0": float, "t1": float, "snap": float, "whistle": float}]
    """

    duration = _video_duration(video_path)
    if duration <= 0:
        return []

    wav = _ensure_wav(video_path)
    audio_peaks = audio_rms_peaks(wav)
    try:
        os.unlink(wav)
    except Exception:
        pass

    times, activity = motion_activity_times(video_path)
    windows = find_play_windows((times, activity), audio_peaks, min_play_length, max_play_length, min_play_gap)

    segments: List[Dict] = []
    for idx, (t0, t1, snap, whistle) in enumerate(windows, 1):
        t0 = max(0.0, t0 - preroll)
        t1 = min(duration, t1 + postroll)
        if t1 - t0 > max_play_length:
            t1 = t0 + max_play_length
        segments.append({"id": f"PLAY_{idx:03d}", "t0": t0, "t1": t1, "snap": snap, "whistle": whistle})

    segments.sort(key=lambda s: s["t0"])

    # remove overlaps
    cleaned: List[Dict] = []
    for seg in segments:
        if cleaned and seg["t0"] < cleaned[-1]["t1"]:
            seg["t0"] = cleaned[-1]["t1"]
        if seg["t1"] > duration:
            seg["t1"] = duration
        if seg["t1"] - seg["t0"] >= min_play_length:
            cleaned.append(seg)
    if not cleaned:
        cleaned = [{"id": "PLAY_001", "t0": 0.0, "t1": min(duration, max_play_length), "snap": 0.0, "whistle": min(duration, max_play_length)}]
    return cleaned
