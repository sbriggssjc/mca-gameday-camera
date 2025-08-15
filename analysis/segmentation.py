from __future__ import annotations
from typing import List, Dict
from dataclasses import dataclass
import cv2, numpy as np


@dataclass
class Segment:
    start_ts: float
    end_ts: float

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts


def segment_video(
    path: str,
    min_play_gap: float = 1.5,
    min_play_length: float = 6.0,
    warmup: float = 0.5,
    tail_margin: float = 1.5,
    downscale: int = 2,
    motion_thresh: float = 8.0,   # motion energy threshold (tunable)
    min_active_sec: float = 1.0,  # need at least this much contiguous activity to start a play
) -> List[Dict]:
    """
    Simple motion-based play segmentation.
    Returns: [{"id": "PLAY_001", "t0": start_sec, "t1": end_sec}, ...]
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
        if downscale > 1:
            frame = cv2.resize(frame, (W // downscale, H // downscale))
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5,5), 0)
        return g

    prev = read_gray()
    if prev is None:
        cap.release()
        return []

    motion = []
    idx = 1
    while True:
        g = read_gray()
        if g is None:
            break
        d = cv2.absdiff(g, prev)
        e = float(np.mean(d))
        motion.append(e)
        prev = g
        idx += 1
    cap.release()

    if not motion:
        return []

    # Smooth motion energy
    m = np.array(motion, dtype=np.float32)
    k = max(3, int(0.25 * fps))  # ~0.25s window
    kernel = np.ones(k, dtype=np.float32) / k
    sm = np.convolve(m, kernel, mode="same")

    # Active mask
    active = sm > motion_thresh

    # Convert to segments in seconds with gap/length constraints
    segs: List[Dict] = []
    i = 0
    t = lambda fi: max(0.0, fi / fps)
    last_end = -1e9
    play_idx = 1
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            # continuous active region [i, j)
            t0 = t(i) - warmup
            t1 = t(j) + tail_margin
            # merge by min_gap
            if segs and (t0 - segs[-1]["t1"]) < min_play_gap:
                # extend previous
                segs[-1]["t1"] = t1
            else:
                segs.append({"id": f"PLAY_{play_idx:03d}", "t0": max(0.0, t0), "t1": t1})
                play_idx += 1
            i = j
        else:
            i += 1

    # enforce min length
    segs = [s for s in segs if (s["t1"] - s["t0"]) >= min_play_length]

    # clamp to video duration
    dur = total / fps if total else None
    if dur:
        for s in segs:
            s["t1"] = min(s["t1"], dur)

    return segs
