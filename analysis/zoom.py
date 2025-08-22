"""Auto-zoom helpers."""
from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence, Tuple


def compute_transforms(
    centers_per_frame: Sequence[Sequence[Tuple[float, float]]],
    frame_size: Tuple[int, int],
    padding: float = 0.25,
    alpha: float = 0.2,
) -> List[np.ndarray]:
    """Return per-frame affine transforms that pan/zoom to keep centers visible.

    This is intentionally lightweight: we compute the bounding box of player
    centers each frame, expand it by ``padding`` and smooth the crop region
    with an exponential moving average (``alpha``).  The transform is returned
    as a 2x3 matrix suitable for :func:`cv2.warpAffine`.
    """
    W, H = frame_size
    cx, cy, scale = W / 2.0, H / 2.0, 1.0
    transforms: List[np.ndarray] = []
    for centers in centers_per_frame:
        if centers:
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            pad = padding * max(W, H)
            minx = max(0.0, minx - pad)
            miny = max(0.0, miny - pad)
            maxx = min(W, maxx + pad)
            maxy = min(H, maxy + pad)
            box_w = maxx - minx
            box_h = maxy - miny
            cx = alpha * (minx + box_w / 2.0) + (1 - alpha) * cx
            cy = alpha * (miny + box_h / 2.0) + (1 - alpha) * cy
            if box_w > 0 and box_h > 0:
                scale = alpha * min(W / box_w, H / box_h) + (1 - alpha) * scale
        M = np.array([[scale, 0, -cx * scale + W / 2.0], [0, scale, -cy * scale + H / 2.0]])
        transforms.append(M)
    return transforms
