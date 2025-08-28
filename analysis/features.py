from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional heavy deps
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None  # type: ignore


def _coarse_from_players(players: List[Dict[str, Any]], W: int = 1280, H: int = 720) -> Optional[Dict[str, float]]:
    """Build a coarse feature vector from raw player detections."""

    xs: List[float] = []
    ys: List[float] = []
    for p in players:
        if "x" in p and "y" in p:
            xs.append(float(p["x"]))
            ys.append(float(p["y"]))
        elif "bbox" in p and len(p["bbox"]) >= 4:
            x1, y1, x2, y2 = p["bbox"][:4]
            xs.append(0.5 * (x1 + x2))
            ys.append(y2)
    n = len(xs)
    if n == 0:
        return None

    W = float(max(W, 1))
    H = float(max(H, 1))
    xn = [x / W for x in xs]
    yn = [y / H for y in ys]
    mx = sum(xn) / n
    my = sum(yn) / n
    sx = (sum((x - mx) ** 2 for x in xn) / max(1, n - 1)) ** 0.5 if n > 1 else 0.0
    sy = (sum((y - my) ** 2 for y in yn) / max(1, n - 1)) ** 0.5 if n > 1 else 0.0
    spread_x = (max(xn) - min(xn)) if n > 1 else 0.0
    spread_y = (max(yn) - min(yn)) if n > 1 else 0.0
    return {
        "n_players": n,
        "mx": mx,
        "sx": sx,
        "my": my,
        "sy": sy,
        "spread_x": spread_x,
        "spread_y": spread_y,
    }


def compute_all(tracks: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None, min_players: int = 3) -> List[Dict[str, Any]]:
    """Compute coarse features for all segments.

    Always returns a row per input segment with a ``reason`` field when
    features cannot be computed.
    """

    W = (meta or {}).get("width", 1280)
    H = (meta or {}).get("height", 720)
    feats: List[Dict[str, Any]] = []
    for t in tracks:
        sid = t.get("segment_id") or t.get("seg_id")
        players = t.get("players", [])
        coarse = _coarse_from_players(players, W, H)
        if coarse is None:
            feats.append({
                "segment_id": sid,
                "num_players": 0,
                "features": {},
                "reason": "no_tracks",
            })
            continue
        feat_dict = {
            "mx": coarse["mx"],
            "sx": coarse["sx"],
            "my": coarse["my"],
            "sy": coarse["sy"],
            "spread_x": coarse["spread_x"],
            "spread_y": coarse["spread_y"],
        }
        reason = "ok" if coarse["n_players"] >= min_players else "low_players"
        feats.append({
            "segment_id": sid,
            "num_players": coarse["n_players"],
            "features": feat_dict,
            "reason": reason,
        })
    return feats


# ---------------------------------------------------------------------------
# New audio + motion helpers for segmentation


def audio_rms_peaks(
    audio_path: str,
    sr: int = 16000,
    hop_ms: int = 20,
    smooth_ms: int = 200,
    zscore: float = 2.5,
) -> List[float]:
    """Return list of seconds corresponding to likely whistles / bursts of sound."""

    if librosa is None:
        return []
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception:
        return []
    hop = max(1, int(sr * hop_ms / 1000))
    frame = hop * 2
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    smooth = max(1, int(smooth_ms / hop_ms))
    kernel = np.ones(smooth, dtype=np.float32) / float(smooth)
    sm = np.convolve(rms, kernel, mode="same")
    if sm.size < 3:
        return []
    z = (sm - sm.mean()) / (sm.std() + 1e-6)
    peaks: List[float] = []
    for i in range(1, len(z) - 1):
        if z[i] > zscore and z[i] > z[i - 1] and z[i] > z[i + 1]:
            peaks.append(i * hop / sr)
    return peaks


def motion_activity_times(
    video_path: str,
    step: int = 2,
    win: int = 15,
    zscore: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return time stamps and normalized activity curve using sparse optical flow."""

    if cv2 is None:
        return np.zeros(0), np.zeros(0)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros(0), np.zeros(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return np.zeros(0), np.zeros(0)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    mags: List[float] = []
    times: List[float] = []
    idx = 1
    while True:
        for _ in range(step - 1):
            cap.grab()
            idx += 1
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(float(np.mean(mag)))
        times.append(idx / fps)
        prev = gray
        idx += 1
    cap.release()
    if not mags:
        return np.zeros(0), np.zeros(0)
    arr = np.array(mags, dtype=np.float32)
    k = max(1, int(win))
    kernel = np.ones(k, dtype=np.float32) / float(k)
    sm = np.convolve(arr, kernel, mode="same")
    z = (sm - sm.mean()) / (sm.std() + 1e-6)
    return np.array(times, dtype=np.float32), z


def find_play_windows(
    activity_curve: Tuple[np.ndarray, np.ndarray],
    audio_peaks: List[float],
    min_len: float,
    max_len: float,
    min_gap: float,
) -> List[Tuple[float, float, float, float]]:
    """Fuse motion activity and audio peaks into play windows.

    Returns list of (t0, t1, snap, whistle) tuples."""

    times, vals = activity_curve
    if len(times) == 0:
        return []
    active = vals > 0
    segs: List[Tuple[float, float, float, float]] = []
    i = 0
    n = len(active)
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            t0 = float(times[i])
            tend = float(times[min(j, n - 1)])
            snap = t0
            whistle = tend
            for p in audio_peaks:
                if p >= tend:
                    whistle = p
                    break
            t1 = whistle
            if (t1 - t0) < min_len:
                t1 = t0 + min_len
            if (t1 - t0) > max_len:
                # split long window into chunks of max_len
                start = t0
                while start < t1:
                    end = min(start + max_len, t1)
                    segs.append((start, end, start, min(end, whistle)))
                    start = end
            else:
                segs.append((t0, t1, snap, whistle))
            i = j
        else:
            i += 1

    # enforce min_gap by merging small gaps
    merged: List[Tuple[float, float, float, float]] = []
    for seg in segs:
        if merged and seg[0] - merged[-1][1] < min_gap:
            prev = merged[-1]
            merged[-1] = (prev[0], seg[1], prev[2], seg[3])
        else:
            merged.append(seg)
    return merged



