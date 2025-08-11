#!/usr/bin/env python3
import argparse, json, math, os
from pathlib import Path
import cv2
import numpy as np

def segments_from_motion(video:str, threshold:float, min_seg:float, merge_gap:float=0.8):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if frames > 0 else (cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0)
    step = max(1, int(round(fps / 5.0)))  # sample about 5 fps
    prev = None
    scores = []
    t_stamps = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok: break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (9,9), 0)
            if prev is not None:
                diff = cv2.absdiff(gray, prev).astype(np.float32)
                score = float(diff.mean() / 255.0)
                scores.append(score)
                t_stamps.append(idx / fps)
            prev = gray
        idx += 1
    cap.release()

    # threshold -> contiguous segments
    segs = []
    i = 0
    n = len(scores)
    while i < n:
        if scores[i] > threshold:
            start = t_stamps[i]
            j = i + 1
            while j < n and scores[j] > threshold:
                j += 1
            end = t_stamps[j-1] if j-1 < len(t_stamps) else t_stamps[-1]
            if end - start >= min_seg:
                segs.append([start, end])
            i = j
        else:
            i += 1

    # merge close segments
    merged = []
    for s, e in segs:
        if not merged: merged.append([s,e]); continue
        ps, pe = merged[-1]
        if s - pe < merge_gap:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s,e])
    # Clamp duration
    if duration and duration > 0:
        for r in merged:
            r[0] = max(0.0, r[0]); r[1] = min(duration, r[1])
    return merged, duration

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="OUT folder (will write plays.jsonl here)")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-seg-sec", type=float, default=2.0)
    ap.add_argument("--pad-before", type=float, default=0.6)
    ap.add_argument("--pad-after", type=float, default=1.2)
    args = ap.parse_args()

    segs, duration = segments_from_motion(args.video, args.threshold, args.min_seg_sec)
    # pad + clamp
    padded = []
    for s,e in segs:
        s2 = max(0.0, s - args.pad_before)
        e2 = e + args.pad_after
        if duration: e2 = min(duration, e2)
        if e2 > s2:
            padded.append([s2,e2])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plays_path = out / "plays.jsonl"
    with plays_path.open("w", encoding="utf-8") as f:
        for i,(s,e) in enumerate(padded, 1):
            f.write(json.dumps({"play_id": i, "start_s": round(s,3), "end_s": round(e,3), "label":"UNKNOWN", "source":"fallback_motion"})+"\n")

    print(f"fallback_motion segments: {len(padded)}")
    print(f"wrote: {plays_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
