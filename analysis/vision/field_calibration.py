"""Field calibration utilities.

This module supports computing a homography between image pixels and
field coordinates.  Calibration is stored as JSON with ``image_points``
(4 corner clicks in pixels) and ``field_points`` (their corresponding
coordinates in yards).  The JSON file can be generated once offline and
reused for subsequent runs.
"""
from __future__ import annotations

import json
import os
from typing import Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


class FieldCalibrator:
    """Load and apply a homography for the playing field."""

    def __init__(self, calib_path: str | None = None) -> None:
        self.calib_path = calib_path
        self.h: np.ndarray | None = None
        if calib_path and os.path.exists(calib_path):
            self.load(calib_path)

    # ------------------------------------------------------------------
    def load(self, path: str) -> None:
        """Load calibration data from ``path``."""

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        src = np.array(data["image_points"], dtype=np.float32)
        dst = np.array(data["field_points"], dtype=np.float32)
        self.h, _ = cv2.findHomography(src, dst)

    # ------------------------------------------------------------------
    def pixel_to_field(self, pt: Point) -> Point | None:
        if self.h is None:
            return None
        x, y = pt
        vec = np.array([x, y, 1.0], dtype=np.float32)
        dst = self.h @ vec
        return float(dst[0] / dst[2]), float(dst[1] / dst[2])

    # ------------------------------------------------------------------
    def field_to_pixel(self, pt: Point) -> Point | None:
        if self.h is None:
            return None
        x, y = pt
        h_inv = np.linalg.inv(self.h)
        vec = np.array([x, y, 1.0], dtype=np.float32)
        dst = h_inv @ vec
        return float(dst[0] / dst[2]), float(dst[1] / dst[2])


def compute_homography(
    image_points: Sequence[Point], field_points: Sequence[Point]
) -> np.ndarray:
    """Return a homography matrix for the provided correspondences."""

    src = np.array(list(image_points), dtype=np.float32)
    dst = np.array(list(field_points), dtype=np.float32)
    h, _ = cv2.findHomography(src, dst)
    return h
