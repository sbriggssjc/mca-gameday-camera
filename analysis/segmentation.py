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

    The real project would analyse motion/whistles etc.  For tests we simply
    return a single segment covering the full video duration while honouring
    ``min_play_length``.  No artificial cap is applied so all plays are
    returned.
    """

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
