# Placeholder tracker with motion-blob fallback
from __future__ import annotations

import cv2  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any

# --- Motion-blob fallback helpers ---
ENABLE_MOTION_BLOB_FALLBACK = True
FALLBACK_MIN_AREA = 180  # reject tiny blobs
FALLBACK_FRAME_STEP = 2  # analyze every Nth frame
FALLBACK_CONFIDENCE = 0.40
FALLBACK_MAX_BLOBS = 16


def _fallback_motion_blobs(
    vcap,
    t0_s: float,
    t1_s: float,
    frame_step: int = FALLBACK_FRAME_STEP,
    min_area: int = FALLBACK_MIN_AREA,
) -> list:
    """Lightweight motion-blob detector using background subtraction.

    Returns per-frame dicts: {"ts": float, "boxes": [[x1,y1,x2,y2,conf], ...]}
    """

    vcap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t0_s) * 1000.0)
    H = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    W = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    if H <= 0 or W <= 0:
        return []

    bs = cv2.createBackgroundSubtractorMOG2(
        history=250, varThreshold=16, detectShadows=False
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    frames = []
    fidx = 0
    while True:
        ok, frame = vcap.read()
        if not ok:
            break
        ts = (vcap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if ts > t1_s:
            break
        if (fidx % frame_step) != 0:
            fidx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        fg = bs.apply(gray)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.dilate(fg, kernel, iterations=1)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < min_area:
                continue
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2, float(FALLBACK_CONFIDENCE)])
            if len(boxes) >= FALLBACK_MAX_BLOBS:
                break

        frames.append({"ts": ts, "boxes": boxes})
        fidx += 1

    return frames


def _frames_to_tracking_row(seg_id: str, frames: list, used_fallback: bool) -> dict:
    players = []
    for fr in frames:
        ts = fr["ts"]
        for x1, y1, x2, y2, conf in fr.get("boxes", []):
            players.append({"ts": ts, "bbox": [x1, y1, x2, y2], "conf": conf})
    reason = "no_detections" if len(players) == 0 else "ok"
    return {
        "segment_id": seg_id,
        "players": players,
        "meta": {"used_fallback": bool(used_fallback), "frames": len(frames)},
        "reason": reason,
    }

# --- end helpers ---


def track(video_path: str, segments: List[Dict[str, Any]], team: str | None = None, team_color: str | None = None) -> List[Dict[str, Any]]:
    vcap = cv2.VideoCapture(video_path)
    rows: List[Dict[str, Any]] = []
    for seg in segments:
        seg_id = seg.get("segment_id") or seg.get("id")
        t0 = float(seg.get("start_s") or seg.get("start") or 0.0)
        t1 = float(seg.get("end_s") or seg.get("end") or (t0 + 6.0))

        # Primary detector path (placeholder)
        primary_frames: List[Dict[str, Any]] = []
        try:
            # Example: primary_frames = detector.track_segment(vcap, t0, t1)
            pass
        except Exception as e:  # pragma: no cover - best effort
            print(f"[tracker] primary detector error for {seg_id}: {e}")

        total_primary = sum(len(fr.get("boxes", [])) for fr in primary_frames)

        if ENABLE_MOTION_BLOB_FALLBACK and total_primary < 5:
            vcap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
            fb_frames = _fallback_motion_blobs(vcap, t0, t1)
            row = _frames_to_tracking_row(seg_id, fb_frames, used_fallback=True)
        else:
            row = _frames_to_tracking_row(seg_id, primary_frames, used_fallback=False)

        rows.append(row)

    vcap.release()
    return rows
