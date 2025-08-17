import numpy as np

try:  # pragma: no cover
    import cv2
except Exception:  # pragma: no cover - graceful fallback if cv2 missing
    cv2 = None


def estimate_rotation_degrees(video_path: str, sample_frames: int = 30) -> float:
    """Estimate dominant field rotation in degrees.

    We sample a handful of frames, run edge/line detection, and compute the
    median angle of strong lines.  The result is snapped to the nearest of
    {0, 90, 180, 270} when within ~15 degrees.  Falls back to a simple aspect
    ratio heuristic when OpenCV or the video cannot be read.
    """
    if cv2 is None:
        return 0.0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        total = sample_frames
    idxs = np.linspace(0, max(0, total - 1), num=min(sample_frames, total))
    angles: list[float] = []
    w = h = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
        if lines is None:
            continue
        for rho, theta in lines[:20]:
            ang = theta * 180.0 / np.pi
            ang = (ang + 90) % 180 - 90  # map to [-90, 90)
            angles.append(float(ang))
    cap.release()
    if not angles:
        # Fallback: portrait vs landscape
        if h > w:
            return 90.0
        return 0.0
    med = float(np.median(angles))
    candidates = [0.0, 90.0, 180.0, 270.0]
    best = min(candidates, key=lambda c: abs((med - c + 180) % 360 - 180))
    if abs(best - med) <= 15.0:
        med = best
    return float(med % 360)


def normalize_orientation(frame, rotate_deg: int):
    """Rotate frame by ``rotate_deg`` degrees clockwise."""
    if cv2 is None or rotate_deg % 360 == 0:
        return frame
    rot = rotate_deg % 360
    if rot == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # arbitrary angles
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot, 1.0)
    return cv2.warpAffine(frame, M, (w, h))
