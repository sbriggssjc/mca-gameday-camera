from __future__ import annotations
import numpy as np, json
from pathlib import Path

try:  # optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - missing libs
    cv2 = None

def infer_generic(fr_dict):
    """
    Extremely simple spacing-based formation buckets:
      - 'spread' : >=3 wide alignments (players near numbers)
      - 'balanced' : 2 wide each side roughly symmetric
      - 'tight' : most blobs near box
    This is intentionally coarse and jersey/color-agnostic.
    """
    # expects precomputed 'player_x' normalized positions if available
    xs = fr_dict.get("player_x_norm", [])
    if not xs: return "unknown"
    xs = [x for x in xs if 0.0 <= x <= 1.0]
    if len(xs) < 7:  # low detections
        return "unknown"

    left = sum(1 for x in xs if x < 0.2)
    right = sum(1 for x in xs if x > 0.8)
    wide = left + right
    mid = sum(1 for x in xs if 0.4 < x < 0.6)

    if wide >= 3:
        return "spread"
    if mid >= len(xs) * 0.5:
        return "tight"
    return "balanced"
