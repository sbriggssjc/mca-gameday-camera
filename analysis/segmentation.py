from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    """Represents a play segment in the source video."""

    start_ts: float
    end_ts: float

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts


def segment_video(*args, **kwargs):
    raise NotImplementedError("Legacy segment_video disabled. Use SnapWhistleFinder.find_plays().")


def windowize_segments(*args, **kwargs):
    raise NotImplementedError("Legacy windowize_segments disabled. Snap→Whistle produces final windows.")
