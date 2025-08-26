from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import subprocess, json, shlex

try:  # pragma: no cover
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def _probe_fps_ffprobe(path: str) -> float:
    try:
        cmd = (
            f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate '
            f'-of json {shlex.quote(path)}'
        )
        out = subprocess.check_output(cmd, shell=True).decode("utf-8", "ignore")
        data = json.loads(out)
        rate = data["streams"][0].get("r_frame_rate", "0/1")
        num, den = rate.split("/")
        num, den = float(num), float(den)
        return num / den if den else 0.0
    except Exception as e:
        print(f"[ffprobe] failed to probe fps: {e}")
        return 0.0


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
    """Simple motion-based play segmentation.

    When OpenCV is unavailable, we fall back to returning a single segment
    spanning the entire video duration (via ``ffprobe``).
    """
    if cv2 is None:
        try:
            out = subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=duration", "-of", "json", path
            ])
            dur = float(json.loads(out)["streams"][0].get("duration", 0.0))
        except Exception:
            dur = 0.0
        return [{"id": "PLAY_001", "t0": 0.0, "t1": max(10.0, dur)}]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    if not fps or fps < 5 or fps > 240:
        print(f"[video] OpenCV fps={fps} looks wrong; reprobing via ffprobe…")
        fps2 = _probe_fps_ffprobe(path)
        if fps2 > 1:
            print(f"[video] using ffprobe fps={fps2}")
            fps = fps2
        else:
            print("[video] ffprobe also failed; keeping OpenCV fps")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

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
    play_idx = 1
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            t0 = t(i) - warmup
            t1 = t(j) + tail_margin
            if segs and (t0 - segs[-1]["t1"]) < min_play_gap:
                segs[-1]["t1"] = t1
            else:
                segs.append({"id": f"PLAY_{play_idx:03d}", "t0": max(0.0, t0), "t1": t1})
                play_idx += 1
            i = j
        else:
            i += 1

    segs = [s for s in segs if (s["t1"] - s["t0"]) >= min_play_length]
    dur = total / fps if total else None
    if dur:
        for s in segs:
            s["t1"] = min(s["t1"], dur)
    return segs
