from __future__ import annotations

import cv2
import json
import numpy as np
from pathlib import Path


def _read_first_frames(path: str | Path, n: int = 6):
    cap = cv2.VideoCapture(str(path))
    frames = []
    for _ in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def _motion_dir(frames: list[np.ndarray]):
    """Average optical flow direction sign across early frames."""
    if len(frames) < 2:
        return 0.0, 0.0
    vx_all: list[float] = []
    vy_all: list[float] = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for cur in frames[1:]:
        g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        f = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vx_all.append(float(np.mean(f[..., 0])))
        vy_all.append(float(np.mean(f[..., 1])))
        prev = g
    return float(np.mean(vx_all or [0])), float(np.mean(vy_all or [0]))


def infer_side_by_possession(mp4_path: str, black_ratio: float, white_ratio: float):
    """Infer offensive/defensive side using early motion and color ratios.

    Heuristic:

    - If color calibration says black dominates and early net motion is forward,
      assume black offense.
    - Otherwise if white dominates with motion, assume black defense.
    - If ratios are close or motion is low, return ``"unknown"``.
    """

    frames = _read_first_frames(mp4_path, n=6)
    vx, vy = _motion_dir(frames)
    speed = abs(vx) + abs(vy)
    if speed < 0.02:
        return "unknown", 0.40
    if black_ratio > white_ratio * 1.05:
        return "offense", 0.70
    if white_ratio > black_ratio * 1.05:
        return "defense", 0.70
    return "unknown", 0.40


