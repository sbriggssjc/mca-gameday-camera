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
from typing import Sequence, Tuple, List, Optional

import numpy as np

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
    frame: np.ndarray,
    *,
    headless: bool = False,
    points: Optional[Sequence[Point]] = None,
    save_to: str = DEFAULT_CALIB_PATH,
) -> FieldCalibrator:
    """Compute and save a field homography.

    ``frame`` is the image to calibrate.  If ``points`` is provided, it must be
    a list of four ``(x, y)`` pixel coordinates (clockwise starting near the
    left goal line corner) and no GUI will be shown.  If ``points`` is ``None``
    and ``headless`` is ``True``, the frame is written to ``configs/calib_frame.jpg``
    and a ``RuntimeError`` is raised instructing the caller how to proceed.  If
    neither ``points`` nor ``headless`` are provided, an interactive OpenCV
    window is used to collect the clicks.
    """

    image_points: List[Point]

    if points is not None:
        image_points = list(points)
    else:
        if headless:
            import cv2  # Lazy import to keep tests light

            calib_frame = os.path.join("configs", "calib_frame.jpg")
            os.makedirs(os.path.dirname(calib_frame), exist_ok=True)
            cv2.imwrite(calib_frame, frame)
            raise RuntimeError(
                "Run with --points 'x1,y1 x2,y2 x3,y3 x4,y4' or copy configs/calib_frame.jpg to a GUI host to click points."
            )

        import cv2  # Imported lazily to avoid dependency for tests
        import sys

        # If there's no display, OpenCV will fail.  Provide an actionable error
        if sys.platform != "win32" and os.environ.get("DISPLAY") in (None, ""):
            raise RuntimeError(
                "No display available for calibration. Set DISPLAY or run with --headless."
            )

        pts: List[Point] = []

        def on_click(event, x, y, _flags, _param) -> None:
            nonlocal pts
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                pts.append((float(x), float(y)))
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("calibrate", frame)

        clone = frame.copy()
        cv2.imshow("calibrate", clone)
        cv2.setMouseCallback("calibrate", on_click)
        while len(pts) < 4:
            cv2.waitKey(10)
        cv2.destroyWindow("calibrate")
        image_points = pts

    h, h_inv = compute_homography(image_points, FIELD_POINTS)

    data = {
        "image_points": [list(p) for p in image_points],
        "field_points": [list(p) for p in FIELD_POINTS],
        "H": h.tolist(),
        "H_inv": h_inv.tolist(),
        "field_dims": {"length": 120.0, "width": 53.3},
    }

    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    with open(save_to, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    return FieldCalibrator(calib_path=save_to, h=h)

