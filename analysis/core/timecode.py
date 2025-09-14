"""Utilities for converting between frames and timestamps."""
from __future__ import annotations

from typing import Iterable, List, Tuple


def frame_to_time(frame: int, fps: float) -> float:
    """Return seconds for ``frame`` at ``fps``."""
    return frame / fps


def time_to_frame(time_s: float, fps: float) -> int:
    """Return frame index for ``time_s`` at ``fps``."""
    return int(round(time_s * fps))


def gap_windows(times: Iterable[float], max_gap: float) -> List[Tuple[float, float]]:
    """Return contiguous windows splitting at gaps > ``max_gap``."""
    sorted_times = sorted(times)
    if not sorted_times:
        return []
    windows: List[Tuple[float, float]] = []
    start = prev = sorted_times[0]
    for t in sorted_times[1:]:
        if t - prev > max_gap:
            windows.append((start, prev))
            start = t
        prev = t
    windows.append((start, prev))
    return windows


__all__ = ["frame_to_time", "time_to_frame", "gap_windows"]
