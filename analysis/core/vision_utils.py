"""Lightweight vision helpers.

The original project contained small ROI and tracking utilities
scattered across modules.  This file collects thin wrappers
around them for reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class ROI:
    """Simple rectangular region of interest."""

    x: int
    y: int
    w: int
    h: int

    def as_slice(self) -> Tuple[slice, slice]:
        return slice(self.y, self.y + self.h), slice(self.x, self.x + self.w)


def smooth_points(points: Sequence[Tuple[float, float]], alpha: float = 0.5) -> List[Tuple[float, float]]:
    """Return exponentially smoothed ``points``."""
    if not points:
        return []
    smoothed = [points[0]]
    for x, y in points[1:]:
        px, py = smoothed[-1]
        smoothed.append((alpha * x + (1 - alpha) * px, alpha * y + (1 - alpha) * py))
    return smoothed


__all__ = ["ROI", "smooth_points"]
