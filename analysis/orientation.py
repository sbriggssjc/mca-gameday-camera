from __future__ import annotations
import cv2, numpy as np

def _frame_angles(gray) -> list[float]:
    # Canny + Hough
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None or len(lines) == 0:
        return []

    # OpenCV can return shape (N,1,2) or (N,2). Normalize.
    thetas = []
    for ln in lines[:40]:
        try:
            if hasattr(ln, "__len__") and len(ln) >= 1 and hasattr(ln[0], "__len__"):
                # (1,2) container case
                _, theta = ln[0]
            else:
                # (2,) flat case
                _, theta = ln
            thetas.append(float(theta))
        except Exception:
            continue

    # Convert radians to degrees, fold into [0,180)
    degs = [(np.degrees(t) % 180.0) for t in thetas]
    return degs

def estimate_rotation_degrees(path: str) -> int:
    """
    Return one of {0, 90, 180, 270}. Defensive against odd Hough outputs.
    Falls back to 0 if we can't infer a dominant orientation.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0

    # Sample up to ~10 frames uniformly across the video (cheap)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_idxs = np.linspace(0, max(0, total - 1), num=min(10, max(1, total // 50)), dtype=int)

    all_angles: list[float] = []
    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # downscale for speed & denoise
        h, w = frame.shape[:2]
        scale = max(1, min(h, w) // 480)
        if scale > 1:
            frame = cv2.resize(frame, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_angles.extend(_frame_angles(gray))

    cap.release()

    if not all_angles:
        return 0

    # Histogram over [0,180) to find dominant orientation (vertical vs horizontal)
    bins = np.arange(0, 181, 5)  # 5° bins
    hist, _ = np.histogram(all_angles, bins=bins)
    dom_center = (bins[np.argmax(hist)] + bins[np.argmax(hist)+1]) / 2.0  # bin center

    # Snap to nearest 90 degrees (0, 90)
    snapped_180 = int(round(dom_center / 90.0) * 90) % 180
    # Map to device rotations {0,90,180,270}
    # Heuristic: if dominant is ~0 -> landscape (0 or 180); ~90 -> portrait (90 or 270).
    # Return 0 for ~0, and 90 for ~90; caller can mirror if needed.
    if snapped_180 in (0, 180):
        return 0
    else:
        return 90
