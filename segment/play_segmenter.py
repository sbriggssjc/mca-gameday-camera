from __future__ import annotations

from typing import Any, Dict, Iterable, List

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def _windowize(duration_sec: float, min_len: float, min_gap: float) -> List[Dict[str, Any]]:
    win = max(4.0, float(min_len))      # allow a bit shorter windows
    gap = max(0.8, float(min_gap))
    step = win + gap
    t = 0.0
    segs: List[Dict[str, Any]] = []
    pid = 1
    # Build windows across the whole video
    while t < duration_sec - 0.5 * win:
        start = max(0.0, t)
        end = min(t + win, duration_sec)
        if end - start >= 0.66 * win:
            segs.append({
                "play_id": pid,
                "start_s": round(float(start), 3),
                "end_s": round(float(end), 3),
                "source": "fallback_windowize",
            })
            pid += 1
        t += step
    return segs


def coalesce_segments(segments: Iterable[Dict[str, Any]], min_gap: float, *, allow_merge_sources: Iterable[str] = ("primary",)) -> List[Dict[str, Any]]:
    """Merge segments separated by less than ``min_gap`` when allowed.

    Segments whose ``source`` is not in ``allow_merge_sources`` are kept
    separate so fallback windows are never merged by accident.
    """

    out: List[Dict[str, Any]] = []
    prev: Dict[str, Any] | None = None
    for s in sorted(segments, key=lambda x: x["start_s"]):
        if (
            prev
            and (s["start_s"] - prev["end_s"] < min_gap)
            and (prev.get("source") in allow_merge_sources)
            and (s.get("source") in allow_merge_sources)
        ):
            prev["end_s"] = max(prev["end_s"], s["end_s"])
        else:
            if prev:
                out.append(prev)
            prev = dict(s)
    if prev:
        out.append(prev)
    return out


def primary_detect(video_path: str, fps: float, cfg: Dict[str, Any], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Placeholder for primary segmentation logic.

    Returns any segments already present in ``ctx`` or an empty list.
    """

    segs = ctx.get("segments")
    return list(segs) if segs else []


def segment_video(
    video_path: str,
    fps: float,
    cfg: Dict[str, Any],
    ctx: Dict[str, Any],
    max_per_seg: int | None = None,
) -> List[Dict[str, Any]]:
    """Return list of segments for ``video_path``."""

    # 1) primary segmentation (your existing logic)
    segs = primary_detect(video_path, fps, cfg, ctx)

    # 2) fallback
    if not segs:
        print("Segmentation fallback: only 0 plays found; windowizing video")
        duration_sec = float(ctx.get("video_length_sec") or 0.0)
        if duration_sec <= 0.0 and cv2 is not None:
            cap = cv2.VideoCapture(str(video_path))
            f = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            F = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration_sec = (f / F) if f and F else 0.0
            cap.release()
        segs = _windowize(
            duration_sec,
            cfg.get("min_play_length", 6.0),
            cfg.get("min_play_gap", 1.5),
        )
        if not segs:
            # As a last resort: split into 4 equal chunks so strict can catch it
            chunk = max(1.0, duration_sec / 4.0)
            segs = [
                {
                    "play_id": i + 1,
                    "start_s": round(i * chunk, 3),
                    "end_s": round(min(duration_sec, (i + 1) * chunk), 3),
                    "source": "fallback_quarter",
                }
                for i in range(4)
            ]
        if max_per_seg and fps > 0:
            max_len_sec = max_per_seg / float(fps)
            clamped: List[Dict[str, Any]] = []
            for s in segs:
                end_s = min(s["end_s"], s["start_s"] + max_len_sec)
                if end_s > s["start_s"]:
                    s["end_s"] = round(end_s, 3)
                    clamped.append(s)
            segs = clamped

    # 3) NEVER merge fallback windows here. Only normalize IDs.
    for i, s in enumerate(segs, start=1):
        s["play_id"] = i

    return segs
