"""Projection utilities for mapping pixel coordinates to field coordinates.

The real system performs sophisticated tracking smoothing; here we expose a
minimal subset sufficient for unit tests.  ``to_field`` applies a homography
matrix and ``smooth_track`` performs a 1D Savitzky–Golay filter when available
and falls back to a simple moving average otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - graceful degradation
    cv2 = None  # type: ignore
import numpy as np
try:  # savgol is optional
    from scipy.signal import savgol_filter  # type: ignore
except Exception:  # pragma: no cover - graceful degradation
    savgol_filter = None  # type: ignore


@dataclass
class TrackPoint:
    frame: int
    x: float
    y: float


def to_field(coords_px: Sequence[Sequence[float]], H: 'np.ndarray') -> 'np.ndarray':
    """Project ``coords_px`` into field coordinates using homography ``H``."""

    pts = np.array(coords_px, dtype=float)
    if cv2 is not None:
        warped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H)
        return warped.reshape(-1, 2)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    warped = pts_h @ H.T
    warped /= warped[:, [2]]
    return warped[:, :2]


def smooth_track(track: Sequence[Tuple[float, float]], window: int = 5) -> List[Tuple[float, float]]:
    """Return a smoothed version of ``track``.

    ``track`` is a sequence of ``(x, y)`` tuples.  The function attempts to use a
    Savitzky–Golay filter; when SciPy is unavailable it falls back to a naive
    moving-average approach.
    """

    if not track:
        return []
    xs, ys = zip(*track)
    if savgol_filter is not None and len(track) >= window:
        try:
            sx = savgol_filter(xs, window_length=window, polyorder=2)
            sy = savgol_filter(ys, window_length=window, polyorder=2)
            return list(zip(sx.tolist(), sy.tolist()))
        except Exception:  # pragma: no cover - fallback
            pass
    # Moving average fallback
    def _ma(vals: Sequence[float]) -> List[float]:
        out: List[float] = []
        for i in range(len(vals)):
            start = max(0, i - window // 2)
            end = min(len(vals), i + window // 2 + 1)
            out.append(sum(vals[start:end]) / (end - start))
        return out
    return list(zip(_ma(xs), _ma(ys)))


def project_and_smooth(points_px: Sequence[Sequence[float]], H: 'np.ndarray', *, window: int = 5) -> List[Tuple[float, float]]:
    """Convenience wrapper combining :func:`to_field` and :func:`smooth_track`."""

    field_pts = to_field(points_px, H)
    return smooth_track(field_pts.tolist(), window=window)
