import numpy as np

try:  # pragma: no cover
    import cv2
except Exception:  # pragma: no cover - graceful fallback if cv2 missing
    cv2 = None


def detect_rotation(frame):
    # Heuristic: if portrait and large pillarbox/bars, rotate to landscape 16:9
    h, w = frame.shape[:2]
    if h > w:
        return 90  # rotate clockwise
    return 0


def normalize_orientation(frame, rotate_deg: int):
    if cv2 is None:
        return frame
    if rotate_deg == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def letterbox_strip(frame):
    # Remove solid bars if obvious
    if cv2 is None:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Simple border detect: scan rows/cols for near-constant lines
    # Keep as a stub; return frame unchanged to be safe
    return frame
