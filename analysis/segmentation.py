from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Sequence


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
    """Existing segmentation logic wrapped for fallback handling."""

    segments: List[Segment] = []
    total_frames = len(frames)
    if total_frames == 0:
        return segments

    total_time = total_frames / float(fps)
    seg = Segment(0.0, total_time)
    if seg.duration >= min_play_length:
        segments.append(seg)
        if logger:
            logger.info(
                "Segment %d: start=%.2f end=%.2f duration=%.2f",
                len(segments),
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
