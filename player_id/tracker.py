"""Simple tracker placeholder."""

from __future__ import annotations
from typing import Iterable, List

from schemas import Tracklet


def track(sequence_frames: Iterable[object]) -> List[Tracklet]:
    """Run detection and tracking on a sequence of frames.

    This placeholder implementation simply returns an empty list.  Real
    detection/tracking would create :class:`Tracklet` instances with frame
    indices, bounding boxes and embeddings.
    """

    _ = sequence_frames  # unused in stub
    return []
