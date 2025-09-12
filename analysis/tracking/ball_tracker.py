"""Lightweight football detector and tracker.

This module implements a very small ball tracking pipeline designed for
press‑box style footage.  The goal is to provide a fast and reasonably
robust tracker that can run on an embedded device without any heavy
models.  The tracker combines simple motion estimation, color/shape
priors and a constant–velocity Kalman filter.  The public API mirrors the
rest of the tracking utilities in this repository – :class:`BallTracker`
exposes an :py:meth:`update` method that consumes a BGR frame and
optionally a region of interest and returns a tuple describing the
current estimate of the ball location.

The tracker maintains three states:

``TRACKING``
    A confident detection was associated with the previous estimate.
``SEARCHING``
    A prediction is returned but the confidence has fallen below the
    threshold.
``LOST``
    No reliable position is available.  After a grace period the tracker
    returns ``None`` but keeps the last known bounding box so that callers
    may continue to crop around the previous location.

The behaviour of the tracker can be tuned through
``configs/tracking.yaml`` which currently supports the following keys:

``min_area``
    Minimum contour area for a candidate region.
``max_area``
    Maximum contour area for a candidate region.
``min_confidence``
    Minimum score required to report the ``TRACKING`` state.
``decay_rate``
    Multiplicative decay applied when no detection is observed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml


class TrackState(str, Enum):
    """Simple enumeration for the tracker state."""

    TRACKING = "TRACKING"
    SEARCHING = "SEARCHING"
    LOST = "LOST"


@dataclass
class TrackerConfig:
    """Configuration loaded from ``configs/tracking.yaml``."""

    min_area: int = 30
    max_area: int = 2000
    min_confidence: float = 0.3
    decay_rate: float = 0.9
    lost_threshold: int = 10
    hsv_brown: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (5, 50, 50),
        (25, 255, 255),
    )
    hsv_white: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (0, 0, 200),
        (180, 40, 255),
    )
    hsv_green: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (35, 50, 50),
        (90, 255, 255),
    )
    hsv_green_dull: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (35, 35, 25),
        (95, 255, 255),
    )


def _load_config(path: str) -> TrackerConfig:
    cfg = TrackerConfig()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:  # pragma: no cover - trivial
            data = yaml.safe_load(fh) or {}
        for key, val in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
    return cfg


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection over union for two ``(x, y, w, h)`` boxes."""

    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / float(union)


class BallTracker:
    """Detect and track the football in a sequence of frames."""

    def __init__(
        self,
        config_path: str = "configs/tracking.yaml",
        proc_scale: float = 0.5,
        **kwargs,
    ) -> None:
        self.cfg = _load_config(config_path)
        self.proc_scale = float(proc_scale)

        # Kalman filter with state (x, y, vx, vy) and measurements (x, y)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        self.prev_gray: Optional[np.ndarray] = None
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_conf: float = 0.0
        self.state: TrackState = TrackState.LOST
        self.lost_frames: int = 0
        self.frame_idx: int = 0
        self.search_roi: Optional[Tuple[int, int, int, int]] = None
        self.reacquire_interval = 15
        self.roi_margin = 50
        self.k3 = np.ones((3, 3), np.uint8)
        self.k5 = np.ones((5, 5), np.uint8)

    # ------------------------------------------------------------------
    # internal helpers
    def _safe_roi(self, img, x, y, w, h):
        import numpy as np

        H, W = img.shape[:2]
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(W, int(x + w))
        y1 = min(H, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return None
        return img[y0:y1, x0:x1]

    def _color_mask(self, bgr):
        """
        Robust color mask for a (white-ish) ball on green field.
        Returns uint8 mask or None if the input is invalid.
        """
        if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
            return None
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            return None

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Field green range (to suppress)
        green = cv2.inRange(hsv, (35, 50, 50), (90, 255, 255))

        # White/bright range (typical youth footballs read bright/tan under sun)
        white = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))

        out = cv2.bitwise_and(white, cv2.bitwise_not(green))
        # Ensure uint8 mask
        if out is None or out.size == 0:
            return None
        if out.dtype != np.uint8:
            out = out.astype(np.uint8)
        return out

    def _motion_mask(self, gray):
        """
        Motion mask vs previous grayscale frame.
        Returns uint8 mask or None during warmup / invalid input.
        """
        if gray is None or not isinstance(gray, np.ndarray) or gray.size == 0:
            return None

        # Warmup or resize: initialize and skip one frame
        if getattr(self, "prev_gray", None) is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
            return None

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray  # update for next call

        blur = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thr = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
        if thr.dtype != np.uint8:
            thr = thr.astype(np.uint8)
        return thr

    def _update_search_roi(
        self, x: int, y: int, w: int, h: int, fw: int, fh: int
    ) -> None:
        pad = self.roi_margin
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, fw)
        y1 = min(y + h + pad, fh)
        self.search_roi = (x0, y0, x1 - x0, y1 - y0)

    # ------------------------------------------------------------------
    def reset_on_snap(self, snap_hint: bool) -> None:
        """Reset the tracker when a new play begins."""

        if not snap_hint:
            return
        self.kalman.statePre[:] = 0
        self.kalman.statePost[:] = 0
        self.prev_gray = None
        self.last_bbox = None
        self.last_conf = 0.0
        self.state = TrackState.SEARCHING
        self.lost_frames = 0

    # ------------------------------------------------------------------
    def update(self, frame):
        """
        Safe, guard-rail update.

        Returns a 6-tuple expected by the pipeline:
          (bx, by, bw, bh, conf, state)

        - If no detection: (0, 0, 0, 0, 0.0, 'no_det' or reason)
        - On detection: (x, y, w, h, conf, 'ok')
        """
        import cv2, numpy as np

        # --- Basic frame validation ---
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return (0, 0, 0, 0, 0.0, "empty_frame")

        h, w = frame.shape[:2]

        # Kernels (create lazily)
        if not hasattr(self, "k3"):
            self.k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if not hasattr(self, "k5"):
            self.k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Planes
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            return (0, 0, 0, 0, 0.0, "cvt_gray_error")

        # Masks (each may be None on warmup or failure)
        m_motion = self._motion_mask(gray)
        m_color  = self._color_mask(frame)

        masks = [m for m in (m_motion, m_color) if m is not None]
        if not masks:
            # Likely warmup or very static scene; skip gracefully
            return (0, 0, 0, 0, 0.0, "warmup_or_no_masks")

        # Start with the first valid mask
        mask = masks[0]
        # Bitwise AND with any other masks that match shape
        for m in masks[1:]:
            if m.shape == mask.shape:
                mask = cv2.bitwise_and(mask, m)

        if mask is None or mask.size == 0:
            return (0, 0, 0, 0, 0.0, "mask_none")
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if cv2.countNonZero(mask) == 0:
            return (0, 0, 0, 0, 0.0, "mask_empty")

        # Morphology with guards
        try:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k5)
        except cv2.error:
            # proceed with raw mask if morphology fails
            pass

        # Contours
        try:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except cv2.error:
            return (0, 0, 0, 0, 0.0, "findContours_error")

        if not cnts:
            return (0, 0, 0, 0, 0.0, "no_contours")

        # Pick largest reasonable blob
        areas = [(cv2.contourArea(c), c) for c in cnts]
        areas.sort(key=lambda x: x[0], reverse=True)
        area, best = areas[0]

        min_area = 5.0
        max_area = 0.05 * w * h
        if area < min_area or area > max_area:
            return (0, 0, 0, 0, 0.0, "area_out_of_range")

        x, y, ww, hh = cv2.boundingRect(best)

        # Confidence heuristic: normalized area clipped to [0.0, 1.0]
        conf = float(max(0.0, min(1.0, area / (0.01 * w * h))))  # 1% of frame ~= conf 1.0

        return (int(x), int(y), int(ww), int(hh), conf, "ok")

