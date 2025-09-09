"""Crop a view centred on the ball using field homography information.

``YardCropper`` operates in field coordinates (yards) and converts the
desired window back into image pixel coordinates via the inverse
homography.  A small amount of temporal smoothing is applied and per
frame movement is limited to avoid jitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .field_calibration import field_to_img


@dataclass
class YardCropper:
    """Compute a crop rectangle around the ball.

    Parameters
    ----------
    H:
        Homography mapping image pixels to field coordinates.  ``None``
        disables the calibration behaviour and the full frame is
        returned.
    yards_window:
        Total horizontal span of the crop in yards (default ``40``).
    aspect:
        Desired aspect ratio of the crop (default ``16/9``).
    smooth:
        Exponential moving average factor for temporal smoothing.
    max_move:
        Maximum allowed per frame movement expressed as a fraction of the
        frame width.
    timeout:
        Number of consecutive frames without a ball position before
        defaulting to a wide "coach" view.
    """

    H: Optional[np.ndarray]
    yards_window: float = 40.0
    aspect: float = 16 / 9
    smooth: float = 0.9
    max_move: float = 0.06
    timeout: int = 30
    snap_yards_window: float = 40.0
    snap_smooth: float = 0.5
    snap_max_move: float = 0.2
    snap_frames: int = 30

    def __post_init__(self) -> None:  # type: ignore[override]
        self.h_inv: Optional[np.ndarray] = None if self.H is None else np.linalg.inv(self.H)
        self.last: Optional[Tuple[float, float, float, float]] = None
        self.missing = 0
        self.field_width = 53.3
        self.field_length = 120.0
        self.snap_countdown = 0

    # ------------------------------------------------------------------
    def _default_view(self, frame_shape: Tuple[int, int, int]) -> Tuple[float, float, float, float]:
        h, w = frame_shape[:2]
        if w >= 3840 and h >= 2160:
            x = (w - 3840) // 2
            y = (h - 2160) // 2
            return float(x), float(y), 3840.0, 2160.0
        return 0.0, 0.0, float(w), float(h)

    # ------------------------------------------------------------------
    def _compute_target(
        self,
        ball_xy: Optional[Tuple[float, float]],
        frame_shape: Tuple[int, int, int],
        yards_window: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        if ball_xy is None or self.h_inv is None:
            return None

        x, y = ball_xy
        half_w = yards_window / 2.0
        x_min = x - half_w
        x_max = x + half_w
        height_yards = yards_window / self.aspect
        y_min = y - height_yards / 2.0
        y_max = y + height_yards / 2.0

        if x_min < 0:
            x_max = min(self.field_length, x_max - x_min)
            x_min = 0.0
        if x_max > self.field_length:
            x_min = max(0.0, x_min - (x_max - self.field_length))
            x_max = self.field_length

        if y_min < 0:
            y_max = min(self.field_width, y_max - y_min)
            y_min = 0.0
        if y_max > self.field_width:
            y_min = max(0.0, y_min - (y_max - self.field_width))
            y_max = self.field_width

        field_corners = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ]
        img_pts = [field_to_img(pt, self.h_inv) for pt in field_corners]
        xs = [p[0] for p in img_pts]
        ys = [p[1] for p in img_pts]
        h_img, w_img = frame_shape[:2]
        x0 = max(0.0, min(xs))
        x1 = min(float(w_img), max(xs))
        y0 = max(0.0, min(ys))
        y1 = min(float(h_img), max(ys))
        return x0, y0, x1 - x0, y1 - y0

    # ------------------------------------------------------------------
    def compute(
        self,
        frame_shape: Tuple[int, int, int],
        ball_xy: Optional[Tuple[float, float]],
        snap_hint: bool = False,
    ) -> Tuple[int, int, int, int]:
        """Return ``(x, y, w, h)`` crop rectangle for the frame."""

        if snap_hint:
            self.snap_countdown = self.snap_frames
            self.last = None

        current_window = (
            self.snap_yards_window if self.snap_countdown > 0 else self.yards_window
        )
        target = self._compute_target(ball_xy, frame_shape, current_window)
        if target is None:
            self.missing += 1
            if self.last is None or self.missing > self.timeout:
                target = self._default_view(frame_shape)
            else:
                target = self.last
        else:
            self.missing = 0

        use_smooth = self.snap_smooth if self.snap_countdown > 0 else self.smooth
        use_max_move = self.snap_max_move if self.snap_countdown > 0 else self.max_move

        if self.last is None:
            smoothed = target
        else:
            lx, ly, lw, lh = self.last
            tx, ty, tw, th = target
            smoothed = (
                lx * use_smooth + tx * (1 - use_smooth),
                ly * use_smooth + ty * (1 - use_smooth),
                lw * use_smooth + tw * (1 - use_smooth),
                lh * use_smooth + th * (1 - use_smooth),
            )

            max_move_px = frame_shape[1] * use_max_move
            cx_last = lx + lw / 2.0
            cy_last = ly + lh / 2.0
            cx_new = smoothed[0] + smoothed[2] / 2.0
            cy_new = smoothed[1] + smoothed[3] / 2.0
            dx = cx_new - cx_last
            dy = cy_new - cy_last
            if abs(dx) > max_move_px:
                cx_new = cx_last + np.sign(dx) * max_move_px
            if abs(dy) > max_move_px:
                cy_new = cy_last + np.sign(dy) * max_move_px
            smoothed = (
                cx_new - smoothed[2] / 2.0,
                cy_new - smoothed[3] / 2.0,
                smoothed[2],
                smoothed[3],
            )

        x, y, w, h = smoothed
        h_img, w_img = frame_shape[:2]
        x = max(0.0, min(x, w_img - w))
        y = max(0.0, min(y, h_img - h))
        rect = (int(x), int(y), int(w), int(h))
        self.last = rect
        if self.snap_countdown > 0:
            self.snap_countdown -= 1
        return rect

