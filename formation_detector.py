"""Simple offensive formation detector based on player positions."""

from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np


def detect_formation(
    frame_image: np.ndarray,
    player_bboxes: List[Tuple[int, int, int, int]],
    *,
    play_id: int | None = None,
    frame_id: int | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Classify offensive formation from bounding box geometry.

    Parameters
    ----------
    frame_image:
        Full video frame.
    player_bboxes:
        Detected player boxes as ``(x1, y1, x2, y2)``.
    play_id:
        Optional play identifier for the returned info dict.
    frame_id:
        Optional frame identifier for the returned info dict.
    """
    height, width = frame_image.shape[:2]
    if not player_bboxes:
        info = {
            "play_id": play_id,
            "frame_id": frame_id,
            "formation": "Unknown",
            "player_count": 0,
            "spread_strength": "Unknown",
        }
        return "Unknown", info

    centers = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in player_bboxes]
    left = [c for c in centers if c[0] < width / 2]
    right = [c for c in centers if c[0] >= width / 2]

    left_count = len(left)
    right_count = len(right)

    if right_count > left_count:
        spread = "Right-heavy"
    elif left_count > right_count:
        spread = "Left-heavy"
    else:
        spread = "Balanced"

    def clustered(points: List[Tuple[float, float]]) -> bool:
        if len(points) < 3:
            return False
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (max(xs) - min(xs)) < width * 0.1 and (max(ys) - min(ys)) < height * 0.1

    bunch_right = clustered(right)
    bunch_left = clustered(left)

    trips_right = len(right) >= 3 and right_count - left_count >= 2
    trips_left = len(left) >= 3 and left_count - right_count >= 2

    formation = "Unknown"
    if bunch_right:
        formation = "Bunch Right"
    elif bunch_left:
        formation = "Lit"  # treat left bunch similar to tight left set
    elif trips_right:
        formation = "Trips Right"
    elif trips_left:
        formation = "Trips Left"
    else:
        diff = right_count - left_count
        if diff > 1:
            formation = "Reo"
        elif diff < -1:
            formation = "Leo"
        elif diff > 0:
            formation = "Rit"
        elif diff < 0:
            formation = "Lit"
        else:
            formation = "I-Formation"

    info = {
        "play_id": play_id,
        "frame_id": frame_id,
        "formation": formation,
        "player_count": len(player_bboxes),
        "spread_strength": spread,
    }
    return formation, info


__all__ = ["detect_formation"]
