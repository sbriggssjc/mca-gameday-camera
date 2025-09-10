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
        proc_scale: Optional[float] = None,
        args=None,
    ) -> None:
        self.cfg = _load_config(config_path)
        if proc_scale is not None:
            self.proc_scale = proc_scale
        else:
            # ``args`` may come from a CLI; default to 0.5 if not provided
            self.proc_scale = getattr(args, "proc_scale", 0.5)

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

    # ------------------------------------------------------------------
    # internal helpers
    def _motion_mask(self, gray: np.ndarray) -> np.ndarray:
        # First frame or size change: initialize and return empty mask
        if getattr(self, "prev_gray", None) is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
            return np.zeros_like(gray, dtype=np.uint8)

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 3)
        return mask

    def _color_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brown = cv2.inRange(
            hsv,
            np.array(self.cfg.hsv_brown[0]),
            np.array(self.cfg.hsv_brown[1]),
        )
        white = cv2.inRange(
            hsv,
            np.array(self.cfg.hsv_white[0]),
            np.array(self.cfg.hsv_white[1]),
        )
        green1 = cv2.inRange(
            hsv,
            np.array(self.cfg.hsv_green[0]),
            np.array(self.cfg.hsv_green[1]),
        )
        green2 = cv2.inRange(
            hsv,
            np.array(self.cfg.hsv_green_dull[0]),
            np.array(self.cfg.hsv_green_dull[1]),
        )
        green = cv2.bitwise_or(green1, green2)
        mask = cv2.bitwise_or(brown, white)
        return cv2.bitwise_and(mask, cv2.bitwise_not(green))

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
    def update(
        self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Tuple[int, int, int, int, float, TrackState]]:
        """Process a new frame.

        Parameters
        ----------
        frame:
            BGR image.
        roi:
            Optional ``(x, y, w, h)`` region in which to search.
        """

        self.frame_idx += 1
        h, w = frame.shape[:2]
        if self.proc_scale != 1.0:
            work = cv2.resize(
                frame,
                (int(w * self.proc_scale), int(h * self.proc_scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            work = frame
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        motion_full = self._motion_mask(gray)
        color_full = self._color_mask(work)
        mask_full = cv2.bitwise_and(motion_full, color_full)

        if roi is not None:
            x0, y0, w0, h0 = roi
        elif self.search_roi is not None and self.frame_idx % self.reacquire_interval != 0:
            x0, y0, w0, h0 = self.search_roi
            x1 = min(x0 + w0, frame.shape[1])
            y1 = min(y0 + h0, frame.shape[0])
            w0, h0 = x1 - x0, y1 - y0
        else:
            x0 = y0 = 0
            w0, h0 = frame.shape[1], frame.shape[0]

        sx0 = int(x0 * self.proc_scale)
        sy0 = int(y0 * self.proc_scale)
        sw0 = int(w0 * self.proc_scale)
        sh0 = int(h0 * self.proc_scale)
        sx1 = min(sx0 + sw0, mask_full.shape[1])
        sy1 = min(sy0 + sh0, mask_full.shape[0])
        mask = mask_full[sy0:sy1, sx0:sx1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k3)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox: Optional[Tuple[int, int, int, int]] = None
        best_score: float = 0.0

        prediction = self.kalman.predict()
        px, py = int(prediction[0]), int(prediction[1])
        predicted_bbox = None
        if self.last_bbox is not None:
            lw, lh = self.last_bbox[2], self.last_bbox[3]
            predicted_bbox = (px - lw // 2, py - lh // 2, lw, lh)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.cfg.min_area or area > self.cfg.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h)
            aspect_score = 1.0 - abs(aspect - 1.5) / 1.5
            if aspect_score < 0:
                continue
            peri = cv2.arcLength(c, True)
            roundness = 4 * np.pi * area / (peri * peri + 1e-5)
            score = aspect_score * roundness
            if predicted_bbox is not None:
                score *= 0.5 + 0.5 * _iou(predicted_bbox, (x, y, w, h))
            if score > best_score:
                best_score = score
                best_bbox = (x, y, w, h)

        if best_bbox is not None:
            bx, by, bw, bh = best_bbox
            cx, cy = bx + bw // 2, by + bh // 2
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
            self.kalman.correct(measurement)
            self.last_bbox = (bx, by, bw, bh)
            self.last_conf = min(1.0, max(best_score, self.last_conf))
            self.lost_frames = 0
            self.state = (
                TrackState.TRACKING
                if self.last_conf >= self.cfg.min_confidence
                else TrackState.SEARCHING
            )
            scale_inv = 1.0 / self.proc_scale
            gx = int(bx * scale_inv) + x0
            gy = int(by * scale_inv) + y0
            gw = int(bw * scale_inv)
            gh = int(bh * scale_inv)
            self._update_search_roi(gx, gy, gw, gh, frame.shape[1], frame.shape[0])
            return (
                gx,
                gy,
                gw,
                gh,
                float(self.last_conf),
                self.state,
            )

        # No detection: decay confidence and return prediction if possible
        self.last_conf *= self.cfg.decay_rate
        self.lost_frames += 1
        if self.last_bbox is not None and self.lost_frames <= self.cfg.lost_threshold:
            bx, by, bw, bh = self.last_bbox
            bx = px - bw // 2
            by = py - bh // 2
            self.last_bbox = (bx, by, bw, bh)
            self.state = (
                TrackState.SEARCHING
                if self.last_conf >= self.cfg.min_confidence
                else TrackState.LOST
            )
            if self.state is TrackState.LOST and self.lost_frames > self.cfg.lost_threshold:
                self.search_roi = None
                return None
            scale_inv = 1.0 / self.proc_scale
            gx = int(bx * scale_inv) + x0
            gy = int(by * scale_inv) + y0
            gw = int(bw * scale_inv)
            gh = int(bh * scale_inv)
            self._update_search_roi(gx, gy, gw, gh, frame.shape[1], frame.shape[0])
            return (
                gx,
                gy,
                gw,
                gh,
                float(self.last_conf),
                self.state,
            )

        self.state = TrackState.LOST
        self.search_roi = None
        return None

