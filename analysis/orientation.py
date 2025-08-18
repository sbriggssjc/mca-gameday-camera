from __future__ import annotations
from typing import List
import cv2
import math
import numpy as np


def _resize_long_side(img, long_side_max=720):
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= long_side_max:
        return img
    scale = long_side_max / float(long_side)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _angles_from_houghp(linesp) -> List[float]:
    # linesp: (N,1,4) or (N,4)
    if linesp is None:
        return []
    linesp = np.squeeze(linesp)
    if linesp.ndim == 1 and linesp.size == 4:
        linesp = linesp.reshape(1, 4)
    angles = []
    for x1, y1, x2, y2 in linesp:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if dx == 0.0 and dy == 0.0:
            continue
        ang = math.degrees(math.atan2(dy, dx))  # relative to +x axis
        # Map to [-90, +90] for rotation estimation (ignore direction)
        if ang > 90:
            ang -= 180
        if ang < -90:
            ang += 180
        angles.append(ang)
    return angles


def _angles_from_hough(lines) -> List[float]:
    # lines: (N,1,2) or (N,2) with (rho, theta)
    if lines is None:
        return []
    lines = np.squeeze(lines)
    if lines.ndim == 1 and lines.size == 2:
        lines = lines.reshape(1, 2)
    angles = []
    for rho_theta in lines:
        # tolerate either shape
        if isinstance(rho_theta, (list, tuple, np.ndarray)) and len(rho_theta) >= 2:
            rho, theta = float(rho_theta[0]), float(rho_theta[1])
            # Convert theta (angle of normal) to line angle relative to horizontal
            ang = math.degrees(theta) - 90.0
            if ang > 90:
                ang -= 180
            if ang < -90:
                ang += 180
            angles.append(ang)
    return angles


def _circular_median_deg(angles: List[float]) -> float:
    if not angles:
        return 0.0
    # robust median with wrap-around: rotate angles so median is computed in a stable window
    # Try multiple rotation anchors and pick the tightest spread
    best_med, best_spread = 0.0, 1e9
    for anchor in (-90, -45, 0, 45, 90):
        shifted = [((a - anchor + 180) % 360) - 180 for a in angles]
        med = float(np.median(shifted))
        spread = float(np.median(np.abs(shifted - med)))
        if spread < best_spread:
            best_spread = spread
            best_med = med + anchor
    # normalize to [-180, 180]
    best_med = ((best_med + 180) % 360) - 180
    # also fold to [-90, 90] since rotation symmetry exists
    if best_med > 90:
        best_med -= 180
    if best_med < -90:
        best_med += 180
    return best_med


def _snap_to_right_angle(deg: float) -> int:
    candidates = np.array([-180.0, -90.0, 0.0, 90.0, 180.0], dtype=np.float32)
    idx = int(np.argmin(np.abs(candidates - deg)))
    return int(candidates[idx])


def estimate_rotation_degrees(video_path: str, *, sample_stride: int = 30, max_frames: int = 300) -> int:
    """
    Estimate camera rotation in degrees. Never raises on empty/odd Hough output.
    Returns one of {-180, -90, 0, 90, 180}.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, sample_stride)
    limit = max_frames

    all_angles: List[float] = []
    grabbed = 0
    for idx in range(0, total, step):
        if grabbed >= limit:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        grabbed += 1

        frame = _resize_long_side(frame, 720)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 60, 180, L2gradient=True)

        # Prefer probabilistic Hough
        linesp = cv2.HoughLinesP(edges, 1, np.pi/180.0, threshold=60, minLineLength=gray.shape[1]//6, maxLineGap=gray.shape[1]//30)
        angles = _angles_from_houghp(linesp)

        # Fallback to classic Hough
        if not angles:
            lines = cv2.HoughLines(edges, 1, np.pi/180.0, threshold=100)
            angles = _angles_from_hough(lines)

        # Keep a few strongest angles per frame
        if angles:
            # robustly keep central angles
            med = np.median(angles)
            sel = [a for a in angles if abs(a - med) <= 20.0]
            all_angles.extend(sel)

    cap.release()

    if not all_angles:
        return 0

    median_angle = _circular_median_deg(all_angles)
    return _snap_to_right_angle(median_angle)


def normalize_orientation(frame, rotate_deg: int):
    """Rotate frame by ``rotate_deg`` degrees clockwise."""
    if rotate_deg % 360 == 0:
        return frame
    rot = rotate_deg % 360
    if rot == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot, 1.0)
    return cv2.warpAffine(frame, M, (w, h))

