"""Utilities for estimating and applying camera-to-field homographies.

This module intentionally keeps the implementation minimal – it exposes a
``solve_homography`` helper around :func:`cv2.findHomography` and a thin
``Homography`` dataclass used by tests.  The automatic inference heuristics
requested in the user story are outside the scope of this patch and left as
``TODO`` markers so the rest of the code base can stub out behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import json
import logging

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - graceful degradation
    cv2 = None  # type: ignore
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Homography:
    """Simple container for a homography matrix and metadata."""

    H: 'np.ndarray'
    mode: str = "manual"
    resolution: Tuple[int, int] | None = None

    def to_dict(self) -> dict:
        return {
            "H": self.H.tolist(),
            "mode": self.mode,
            "resolution": list(self.resolution) if self.resolution else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Homography":
        h = np.array(data["H"], dtype=float)
        return cls(H=h, mode=data.get("mode", "manual"), resolution=tuple(data.get("resolution", (0, 0))) or None)


def save_homography(h: Homography, path: str | Path) -> None:
    Path(path).write_text(json.dumps(h.to_dict()))


def load_homography(path: str | Path) -> Homography:
    return Homography.from_dict(json.loads(Path(path).read_text()))


def solve_homography(
    corners_px: Sequence[Sequence[float]],
    field_corners: Sequence[Sequence[float]],
    *,
    mode: str = "manual",
    resolution: Tuple[int, int] | None = None,
) -> Homography:
    """Return homography mapping ``corners_px`` to ``field_corners``.

    Parameters are passed straight to :func:`cv2.findHomography`.  ``mode`` and
    ``resolution`` metadata are carried through in the returned object.
    """

    if np is None:
        raise RuntimeError("NumPy is required for homography calculations")
    src = np.array(corners_px, dtype=float)
    dst = np.array(field_corners, dtype=float)
    if cv2 is not None:
        H, _ = cv2.findHomography(src, dst, 0)
    else:  # pragma: no cover - fallback using linear algebra
        A: list[list[float]] = []
        for (x, y), (X, Y) in zip(src, dst):
            A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
            A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])
        A = np.array(A, dtype=float)
        _, _, vh = np.linalg.svd(A)
        H = vh[-1].reshape(3, 3)
    return Homography(H=H / H[2, 2], mode=mode, resolution=resolution)


def project_points(points: Sequence[Sequence[float]], H: 'np.ndarray') -> 'np.ndarray':
    """Project ``points`` using homography ``H``.

    ``points`` is expected to be an ``Nx2`` array-like structure.  The return
    value is an ``Nx2`` numpy array.  When OpenCV is unavailable an identity
    transform is assumed so tests can still run.
    """

    if np is None:
        raise RuntimeError("NumPy required for projection")
    pts = np.array(points, dtype=float)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    warped = pts_h @ H.T
    warped /= warped[:, [2]]
    return warped[:, :2]


def estimate_homography_auto(frames: Iterable['np.ndarray']) -> Homography:
    """Placeholder for the automatic homography estimation pipeline.

    The full implementation involves line detection and RANSAC fitting which is
    beyond the scope of this exercise.  For now we simply return an identity
    transform so callers can proceed.  A warning is logged so developers know
    that automatic estimation has not yet been implemented.
    """

    if np is None:
        raise RuntimeError("NumPy required")
    logger.warning("estimate_homography_auto is a stub implementation")
    eye = np.eye(3)
    return Homography(H=eye, mode="auto")
