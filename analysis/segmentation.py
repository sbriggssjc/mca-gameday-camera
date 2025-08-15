from __future__ import annotations
from typing import List, Dict, Any
import cv2, numpy as np

def segment_video(
    path: str,
    min_play_gap: float = 1.5,
    min_play_length: float = 6.0,
    warmup: float = 0.5,
    tail_margin: float = 1.5,
    downscale: int = 2,
    motion_thresh: float = 8.0,   # higher -> fewer/tighter segments
    **kwargs: Any,                # tolerate extra args (e.g., cfg)
) -> List[Dict]:
    """
    Returns: [{"id": "PLAY_001", "t0": start_sec, "t1": end_sec}, ...]
    Simple frame-diff motion segmentation; robust to sparse tracking.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def read_gray():
        ok, frame = cap.read()
        if not ok:
            return None
        if downscale > 1 and W > 0 and H > 0:
            frame = cv2.resize(frame, (max(1, W // downscale), max(1, H // downscale)))
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5,5), 0)
        return g

    prev = read_gray()
    if prev is None:
        cap.release()
        return []

    motion = []
    while True:
        g = read_gray()
        if g is None:
            break
        motion.append(float(np.mean(cv2.absdiff(g, prev))))
        prev = g
    cap.release()
    if not motion:
        return []

    # Smooth motion energy (~0.25s window)
    k = max(3, int(0.25 * fps))
    kernel = np.ones(k, dtype=np.float32) / k
    sm = np.convolve(np.array(motion, dtype=np.float32), kernel, mode="same")

    active = sm > motion_thresh
    segs: List[Dict] = []
    i = 0
    to_sec = lambda fi: max(0.0, fi / fps)
    play_idx = 1

    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            t0 = max(0.0, to_sec(i) - warmup)
            t1 = to_sec(j) + tail_margin

            if segs and (t0 - segs[-1]["t1"]) < min_play_gap:
                segs[-1]["t1"] = t1
            else:
                segs.append({"id": f"PLAY_{play_idx:03d}", "t0": t0, "t1": t1})
                play_idx += 1
            i = j
        else:
            i += 1

    # Enforce minimum duration and clamp to video duration
    segs = [s for s in segs if (s["t1"] - s["t0"]) >= min_play_length]
    if total:
        dur = total / fps
        for s in segs:
            s["t1"] = min(s["t1"], dur)

    return segs
