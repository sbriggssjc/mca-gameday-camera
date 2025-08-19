# analysis/orientation.py
from __future__ import annotations
import numpy as np

try:  # optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - handle headless environments
    cv2 = None  # type: ignore

def _frame_angles(gray) -> list[float]:
    """Return candidate line angles (degrees) for a single gray frame (0..180)."""
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None or len(lines) == 0:
        return []
    # Normalize shapes: can be (N,1,2) or (N,2)
    thetas = []
    for ln in lines:
        try:
            if hasattr(ln, "__len__") and len(ln) == 1 and hasattr(ln[0], "__len__"):
                rho, theta = ln[0]
            else:
                rho, theta = ln
            thetas.append(float(theta))
        except Exception:
            continue
    # Convert radians to degrees in [0, 180)
    return [(t * 180.0 / np.pi) % 180.0 for t in thetas]

def estimate_rotation_degrees(path: str) -> int:
    """
    Return one of {0, 90, 180, 270} as a coarse device rotation.
    Defensive across OpenCV builds. Falls back to 0 if uncertain.
    """
    if cv2 is None:
        return 0
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0

    # Sample up to ~12 frames across video
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps   = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    HOPS  = max(1, total // 12) if total > 0 else int(5 * fps)

    angles = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % HOPS == 0:
            ok2, frame = cap.retrieve()
            if not ok2 or frame is None:
                idx += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            angles.extend(_frame_angles(gray))
        idx += 1
    cap.release()

    if not angles:
        return 0

    # Map each angle to closest of the 0/90/180/270 "orientations"
    # (for 180/270 reduce mod 180 first; then lift back to 0/90/180/270)
    bins = np.array([0, 90], dtype=float)
    votes = []
    for a in angles:
        a180 = a % 180.0
        nearest = bins[np.argmin(np.abs(bins - a180))]
        # Lift 0/90 into {0,90,180,270} by also voting the opposite direction
        votes.append(nearest)
        votes.append((nearest + 180.0) % 360.0)

    if not votes:
        return 0
    counts = np.bincount(np.round(np.array(votes) % 360).astype(int))
    if counts.size == 0:
        return 0
    # Collapse to the nearest 90° bucket
    quadrant_votes = {
        0:   counts[[0, 1, 359]].sum() if counts.size > 359 else counts.sum(),
        90:  counts[90]  if counts.size > 90  else 0,
        180: counts[180] if counts.size > 180 else 0,
        270: counts[270] if counts.size > 270 else 0,
    }
    dom = max(quadrant_votes, key=quadrant_votes.get)
    return int(dom % 360)


def normalize_orientation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate ``frame`` by multiples of 90 degrees to correct orientation."""
    if cv2 is None:
        return frame
    r = rotation % 360
    if r == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if r == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if r == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

