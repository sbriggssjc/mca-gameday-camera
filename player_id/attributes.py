"""Attribute extraction heuristics."""

from __future__ import annotations

from typing import Dict, Tuple


def extract_attributes(img_crop, full_frame_bbox: Tuple[int, int, int, int]) -> Dict[str, str]:
    """Extract simple color attributes from a player crop.

    This stub merely returns an empty dictionary.  Real implementations would
    analyze the image to derive cleat/sock/glove colors and size estimates.
    """

    _ = (img_crop, full_frame_bbox)
    return {}
