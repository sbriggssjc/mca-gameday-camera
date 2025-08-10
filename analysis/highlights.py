"""Clip generation helpers."""

from __future__ import annotations

import os
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
