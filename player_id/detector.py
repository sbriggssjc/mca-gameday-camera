"""Stub player detector."""

from __future__ import annotations
from typing import List, Dict


def detect_players(frame) -> List[Dict]:
    """Return bounding boxes for players in a frame.

    The real system would run an object detector.  For the purposes of unit
    tests we simply return an empty list which callers must handle.
    """

    return []
