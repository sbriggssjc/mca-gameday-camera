"""Compute a cropped view centred on the ball.

Given the estimated ball position in pixel coordinates, ``YardCropper``
produces a crop rectangle that spans ``±N`` yards horizontally around the
ball (default 20).  When calibration data is available the yard-to-pixel
conversion is based on the homography; otherwise a simple fraction of the
frame width is used.  Basic temporal smoothing is applied to avoid
jarring jumps.
"""
from __future__ import annotations

from typing import Tuple

from .field_calibration import FieldCalibrator


class YardCropper:
    def __init__(
        self,
        calibrator: FieldCalibrator | None,
        crop_yards: int = 20,
        smooth: float = 0.8,
    ) -> None:
        self.calibrator = calibrator
        self.crop_yards = crop_yards
        self.smooth = smooth
        self.last: Tuple[int, int, int, int] | None = None

    def _yards_to_pixels(self, yards: float, ref_pt: Tuple[int, int]) -> int:
        if self.calibrator and self.calibrator.h is not None:
            fx = self.calibrator.pixel_to_field(ref_pt)
            if fx is not None:
                left = self.calibrator.field_to_pixel((fx[0] - yards, fx[1]))
                right = self.calibrator.field_to_pixel((fx[0] + yards, fx[1]))
                if left and right:
                    return int(abs(right[0] - left[0]))
        # Fallback – assume 10 yards ~ 1/6th of the frame width
        return int(ref_pt[0] * (yards / 30.0))

    def compute(self, frame_shape: Tuple[int, int, int], ball_pt: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Return ``(x, y, w, h)`` crop rectangle for the frame."""

        h, w = frame_shape[:2]
        cx, cy = ball_pt
        width = self._yards_to_pixels(self.crop_yards * 2, (cx, cy))
        width = max(1, min(width, w))
        x = max(0, min(cx - width // 2, w - width))
        crop = (x, 0, width, h)
        if self.last is not None:
            lx, ly, lw, lh = self.last
            alpha = self.smooth
            crop = (
                int(alpha * lx + (1 - alpha) * crop[0]),
                0,
                int(alpha * lw + (1 - alpha) * crop[2]),
                h,
            )
        self.last = crop
        return crop
