"""Simple ball detector and tracker.

The implementation is intentionally lightweight – it uses a Hough circle
transform to detect a potential ball and a small Kalman filter to smooth
the trajectory.  A confidence score (0-1) is returned for each update so
that callers can gracefully fall back to a wider crop when tracking is
uncertain.
"""
from __future__ import annotations

import cv2
import numpy as np


class BallTracker:
    """Detect and track the football in a video stream."""

    def __init__(self) -> None:
        # 4 state variables (x, y, vx, vy) and 2 measurements (x, y)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.last_conf: float = 0.0

    def _detect(self, frame: "cv2.Mat") -> tuple[int, int, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=30,
        )
        if circles is not None:
            x, y, r = circles[0][0]
            conf = min(1.0, r / 30.0)
            return int(x), int(y), float(conf)
        # No detection – return centre with zero confidence
        h, w = frame.shape[:2]
        return w // 2, h // 2, 0.0

    def update(self, frame: "cv2.Mat") -> tuple[int, int, float]:
        """Update the tracker with a new frame.

        Returns ``(x, y, confidence)`` where ``(x, y)`` are pixel
        coordinates of the estimated ball position and ``confidence`` is a
        float in the range ``[0, 1]``.  When no detection is available the
        previous state prediction is returned with the last confidence
        score.
        """

        x, y, conf = self._detect(frame)
        if conf > 0.5:  # good detection
            measurement = np.array([[np.float32(x)], [np.float32(y)]])
            self.kalman.correct(measurement)
            self.last_conf = conf
        prediction = self.kalman.predict()
        px, py = int(prediction[0]), int(prediction[1])
        return px, py, self.last_conf
