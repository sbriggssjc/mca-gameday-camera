from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - best effort
    cv2 = None  # type: ignore


def _windowize(duration_sec: float, min_len: float, min_gap: float) -> List[Dict[str, float]]:
    """Generate sliding windows as fallback segmentation."""
    win = max(3.5, float(min_len))
    gap = max(0.8, float(min_gap))
    step = win + gap
    t = 0.0
    segments: List[Dict[str, float]] = []
    pid = 1
    while t + 0.75 * win < duration_sec:
        start = t
        end = min(t + win, duration_sec)
        if end - start >= 0.66 * win:
            segments.append({"play_id": pid, "start_s": float(start), "end_s": float(end)})
            pid += 1
        t += step
    return segments


def segment_video(
    ctx: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    video_path: str | None = None,
) -> List[Dict[str, float]]:
    """Segment a video into play windows.

    This function expects that a primary detection stage has populated ``ctx``
    with a list of segments under the key ``"segments"``.  If no segments are
    present, a sliding window fallback is used based on the minimum play
    length and gap specified in ``cfg``.
    """

    ctx = ctx or {}
    cfg = cfg or {}
    segs = ctx.get("segments") or []

    if not segs:
        print("Segmentation fallback: only 0 plays found; windowizing video")
        duration_sec = float(ctx.get("duration_sec") or ctx.get("video_length_sec") or 0.0)
        if duration_sec <= 0.0 and video_path and cv2 is not None:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration_sec = (frames / fps) if fps and frames else 0.0
            cap.release()
        segs = _windowize(
            duration_sec,
            cfg.get("min_play_length", 6.0),
            cfg.get("min_play_gap", 1.5),
        )
    return segs
