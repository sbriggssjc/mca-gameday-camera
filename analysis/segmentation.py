from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np


@dataclass
class Segment:
    """Represents a play segment in the source video."""

    start_ts: float
    end_ts: float

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts


def segment_video(
    frames: Sequence[Any],
    fps: float,
    min_play_gap: float = 7.0,
    min_play_length: float = 4.0,
    logger: logging.Logger | None = None,
) -> List[Segment]:
    """Segment ``frames`` into plays.

    Runs the primary segmentation logic first and, if that yields too few
    segments, falls back to a simple windowizer that slices the entire video
    into fixed-length windows.  This keeps the downstream pipeline moving even
    when the sophisticated logic fails to find enough plays.
    """

    segments = _primary_segmentation(frames, fps, min_play_gap, min_play_length, logger)

    MIN_PLAYS = 5
    if len(segments) < MIN_PLAYS:
        if logger:
            logger.warning(
                f"Segmentation fallback: only {len(segments)} plays found; windowizing video"
            )
        segments = windowize_segments(
            total_frames=len(frames),
            fps=fps,
            window_sec=12.0,
            gap_sec=2.0,
        )

    return segments


def _primary_segmentation(
    frames: Sequence[Any],
    fps: float,
    min_play_gap: float,
    min_play_length: float,
    logger: logging.Logger | None = None,
) -> List[Segment]:
    """Improved segmentation using adaptive motion thresholds.

    A basic motion energy approach is used to detect periods of activity.
    The temporal median of frame-to-frame absolute differences determines an
    adaptive threshold.  Detected segments are separated by a short "dead"
    time to avoid rapid re-triggering and sub-4s micro segments are merged
    into their neighbours.
    """

    segments: List[Segment] = []
    total_frames = len(frames)
    if total_frames <= 1:
        return segments

    # ------------------------------------------------------------------
    # Motion energy & adaptive thresholding
    # ------------------------------------------------------------------
    energies: List[float] = []
    for i in range(1, total_frames):
        prev, cur = frames[i - 1], frames[i]
        if prev is None or cur is None:
            energies.append(0.0)
            continue
        diff = np.abs(cur.astype("float32") - prev.astype("float32"))
        energies.append(float(diff.mean()))

    if not energies:
        return segments

    median_energy = float(np.median(energies))
    threshold = max(median_energy * 2.0, 1e-6)

    # ------------------------------------------------------------------
    # Scan for segments with dead-time protection
    # ------------------------------------------------------------------
    dead_frames = int(2.0 * fps)
    start_idx: int | None = None
    last_end = -dead_frames

    def _commit(start: int, end: int) -> None:
        seg = Segment(start / fps, end / fps)
        if seg.duration >= min_play_length:
            segments.append(seg)
        elif segments and seg.duration > 0:
            # merge micro segments into previous if close enough
            prev = segments[-1]
            if seg.start_ts - prev.end_ts <= min_play_gap:
                prev.end_ts = seg.end_ts
            else:
                segments.append(seg)

    for idx, energy in enumerate(energies):
        active = energy > threshold
        if start_idx is None:
            if active and idx - last_end > dead_frames:
                start_idx = idx
        else:
            if not active:
                _commit(start_idx, idx)
                last_end = idx
                start_idx = None

    if start_idx is not None:
        _commit(start_idx, total_frames - 1)

    if logger:
        for i, seg in enumerate(segments, 1):
            logger.info(
                "Segment %d: start=%.2f end=%.2f duration=%.2f",
                i,
                seg.start_ts,
                seg.end_ts,
                seg.duration,
            )

    return segments


def windowize_segments(
    total_frames: int,
    fps: float,
    window_sec: float = 12.0,
    gap_sec: float = 2.0,
) -> List[Segment]:
    """Generate fixed-length segments across the video as a simple fallback."""

    segments: List[Segment] = []
    win = int(window_sec * fps)
    gap = int(gap_sec * fps)
    min_len = int(3.0 * fps)

    start = 0
    while start + min_len < total_frames:
        end = min(start + win, total_frames - 1)
        seg = Segment(start / fps, end / fps)
        segments.append(seg)
        start = end + gap

    return segments
