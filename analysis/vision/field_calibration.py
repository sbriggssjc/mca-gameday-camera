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
import logging
from typing import Sequence, Tuple, List, Optional

import numpy as np

log = logging.getLogger(__name__)

Point = Tuple[float, float]


FIELD_POINTS: List[Point] = [
    (0.0, 0.0),
    (120.0, 0.0),
    (120.0, 53.3),
    (0.0, 53.3),
]

DEFAULT_CALIB_PATH = os.path.join("configs", "field_homography.json")


class FieldCalibrator:
    """Load and apply a homography for the playing field."""

    def __init__(self, calib_path: str | None = None, h: np.ndarray | None = None) -> None:
        self.calib_path = calib_path
        self.h: np.ndarray | None = h
        self.h_inv: np.ndarray | None = None if h is None else np.linalg.inv(h)
        if calib_path and os.path.exists(calib_path):
            self.load(calib_path)

    # ------------------------------------------------------------------
    def load(self, path: str) -> None:
        """Load calibration data from ``path``."""

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "H" in data:
            self.h = np.array(data["H"], dtype=np.float64)
            self.h_inv = np.array(data["H_inv"], dtype=np.float64)
        else:
            src = np.array(data["image_points"], dtype=np.float32)
            dst = np.array(data["field_points"], dtype=np.float32)
            self.h, self.h_inv = compute_homography(src, dst)

    # ------------------------------------------------------------------
    def pixel_to_field(self, pt: Point) -> Point | None:
        if self.h is None:
            return None
        x, y = pt
        vec = np.array([x, y, 1.0], dtype=np.float64)
        dst = self.h @ vec
        return float(dst[0] / dst[2]), float(dst[1] / dst[2])

    # ------------------------------------------------------------------
    def field_to_pixel(self, pt: Point) -> Point | None:
        if self.h_inv is None and self.h is None:
            return None
        if self.h_inv is None:
            self.h_inv = np.linalg.inv(self.h)  # type: ignore[arg-type]
        x, y = pt
        vec = np.array([x, y, 1.0], dtype=np.float64)
        dst = self.h_inv @ vec
        return float(dst[0] / dst[2]), float(dst[1] / dst[2])


def compute_homography(
    image_points: Sequence[Point], field_points: Sequence[Point]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a homography matrix and its inverse."""

    src = np.array(list(image_points), dtype=np.float64)
    dst = np.array(list(field_points), dtype=np.float64)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("Need four source and destination points")
    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
        A.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1].reshape(3, 3)
    h /= h[2, 2]
    h_inv = np.linalg.inv(h)
    return h, h_inv


def img_to_field(pt: Point, h: np.ndarray) -> Point:
    x, y = pt
    vec = np.array([x, y, 1.0], dtype=np.float64)
    dst = h @ vec
    return float(dst[0] / dst[2]), float(dst[1] / dst[2])


def field_to_img(pt: Point, h_inv: np.ndarray) -> Point:
    x, y = pt
    vec = np.array([x, y, 1.0], dtype=np.float64)
    dst = h_inv @ vec
    return float(dst[0] / dst[2]), float(dst[1] / dst[2])


def parse_points_str(s: str) -> List[Point]:
    """Parse a space-separated string of ``x,y`` pairs.

    ``s`` should look like ``"x1,y1 x2,y2 x3,y3 x4,y4"``.  Returns a list of
    four ``(x, y)`` tuples.  Raises ``ValueError`` if the input is malformed.
    """

    parts = s.strip().split()
    if len(parts) != 4:
        raise ValueError("Expected four x,y pairs")
    pts: List[Point] = []
    for part in parts:
        try:
            x_str, y_str = part.split(",")
            pts.append((float(x_str), float(y_str)))
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid point '{part}'") from exc
    return pts


def calibrate_from_clicks(
    frame: np.ndarray | None = None,
    *,
    headless: bool = False,
    points: Optional[Sequence[Point]] = None,
    save_to: str = DEFAULT_CALIB_PATH,
) -> dict:
    """Compute and save a field homography.

    If ``points`` are supplied, they must be four pixel coordinates in
    clockwise order starting at the goal-line corner.  In this case no GUI is
    shown and the homography is computed directly.  When ``headless`` is
    ``True`` but ``points`` are not provided, a ``RuntimeError`` is raised.  If
    neither ``headless`` nor ``points`` are specified, an interactive OpenCV
    window is used to collect the four clicks.  ``frame`` must contain the
    image for interactive calibration; this function never attempts to open a
    camera itself.
    """

    import cv2  # Lazy import inside function

    image_points: np.ndarray

    if points is not None:
        image_points = np.array(points, dtype=np.float32)
    else:
        if headless:
            if frame is not None:
                calib_frame = os.path.join("configs", "calib_frame.jpg")
                os.makedirs(os.path.dirname(calib_frame), exist_ok=True)
                cv2.imwrite(calib_frame, frame)
            raise RuntimeError(
                "Headless calibration requires --points or a pre-saved calib_frame.jpg + manual point extraction."
            )
        if frame is None:
            raise ValueError("frame is required for interactive calibration")

        pts: List[Point] = []

        def on_click(event, x, y, _flags, _param) -> None:
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                pts.append((float(x), float(y)))
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                try:
                    cv2.imshow("calibrate", frame)
                except cv2.error as e:  # pragma: no cover - GUI failure
                    raise RuntimeError(
                        "OpenCV GUI not available—use --headless or xvfb."
                    ) from e

        try:
            cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
            cv2.imshow("calibrate", frame.copy())
        except cv2.error as e:  # pragma: no cover - GUI failure
            raise RuntimeError(
                "OpenCV GUI not available—use --headless or xvfb."
            ) from e
        cv2.setMouseCallback("calibrate", on_click)
        while len(pts) < 4:
            cv2.waitKey(10)
        cv2.destroyWindow("calibrate")
        image_points = np.array(pts, dtype=np.float32)

    field_points = np.array(FIELD_POINTS, dtype=np.float32)
    h, _ = cv2.findHomography(image_points, field_points)
    if h is None:
        raise RuntimeError("cv2.findHomography failed")
    h_inv, _ = cv2.findHomography(field_points, image_points)
    if h_inv is None:
        raise RuntimeError("cv2.findHomography failed for inverse")

    proj = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), h).reshape(-1, 2)
    err = proj - field_points
    rms = float(np.sqrt((err ** 2).sum(axis=1).mean()))

    data = {
        "H": h.tolist(),
        "H_inv": h_inv.tolist(),
        "field": {"length": 120.0, "width": 53.3},
    }

    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    with open(save_to, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    return {"H": h, "H_inv": h_inv, "rms": rms}

