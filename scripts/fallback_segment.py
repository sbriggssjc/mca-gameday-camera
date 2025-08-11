#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import cv2, numpy as np

def sample_motion_scores(video:str, target_fps=5.0):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps/target_fps)))
    prev = None
    scores, times = [], []
    i = 0
    while True:
        ok = cap.grab()
        if not ok: break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok: break
            # downscale for speed & noise reduction
            h, w = frame.shape[:2]
            scale = 320.0 / max(w, 1)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7,7), 0)
            if prev is not None:
                diff = cv2.absdiff(gray, prev).astype(np.float32)
                scores.append(float(diff.mean()/255.0))
                times.append(i / fps)
            prev = gray
        i += 1
    cap.release()
    duration = frames / fps if frames > 0 else (times[-1] if times else 0.0)
    return np.array(scores, dtype=np.float32), np.array(times, dtype=np.float32), float(duration)

def segments_from_scores(scores, times, threshold, min_seg, merge_gap=0.8):
    segs = []
    if len(scores) == 0: return segs
    above = scores > threshold
    i, n = 0, len(scores)
    while i < n:
        if above[i]:
            start = float(times[i])
            j = i + 1
            while j < n and above[j]:
                j += 1
            end = float(times[j-1] if j-1 < len(times) else times[-1])
            if end - start >= min_seg:
                segs.append([start, end])
            i = j
        else:
            i += 1
    # merge close
    merged = []
    for s,e in segs:
        if not merged: merged.append([s,e]); continue
        ps,pe = merged[-1]
        if s - pe < merge_gap:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s,e])
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="OUT folder (writes plays.jsonl here)")
    ap.add_argument("--threshold", type=float, default=-1.0, help="Set <0 to auto-pick")
    ap.add_argument("--min-seg-sec", type=float, default=2.0)
    ap.add_argument("--pad-before", type=float, default=0.6)
    ap.add_argument("--pad-after", type=float, default=1.2)
    ap.add_argument("--target-fps", type=float, default=5.0)
    ap.add_argument("--min-target", type=int, default=8, help="Aim for at least this many segments when auto-picking")
    args = ap.parse_args()

    scores, times, duration = sample_motion_scores(args.video, args.target_fps)
    if scores.size:
        mean, std = float(scores.mean()), float(scores.std())
        mx = float(scores.max()); mn = float(scores.min())
    else:
        mean = std = mx = mn = 0.0

    thr = args.threshold
    if thr < 0:
        # auto: start at mean+2*std, then relax until we reach min-target or 0.10
        thr = max(0.10, mean + 2.0*std)
        order = np.sort(scores)[::-1]
        # if even top-quantiles are low, drop to 85th percentile
        if order.size:
            thr = max(0.10, float(np.quantile(scores, 0.85)))
        # final safety: if still too few segments after initial pass, we’ll iterate below

    def build(threshold):
        segs = segments_from_scores(scores, times, threshold, args.min_seg_sec)
        # pad & clamp
        padded=[]
        for s,e in segs:
            s2 = max(0.0, s - args.pad_before)
            e2 = min(duration, e + args.pad_after) if duration>0 else (e + args.pad_after)
            if e2 > s2: padded.append([s2,e2])
        return padded

    segs = build(thr)
    # If auto and we got too few, relax threshold progressively
    if args.threshold < 0 and len(segs) < args.min_target and scores.size:
        for q in (0.80, 0.75, 0.70, 0.65, 0.60):
            thr_try = float(np.quantile(scores, q))
            if thr_try >= thr: continue
            segs = build(thr_try)
            thr = thr_try
            if len(segs) >= args.min_target: break

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plays_path = out / "plays.jsonl"
    with plays_path.open("w", encoding="utf-8") as f:
        for i,(s,e) in enumerate(segs, 1):
            f.write(json.dumps({"play_id": i, "start_s": round(s,3), "end_s": round(e,3),
                                "label": "UNKNOWN", "source": "fallback_motion"}) + "\n")

    print(f"motion scores: n={scores.size}, min={mn:.3f}, max={mx:.3f}, mean={mean:.3f}, std={std:.3f}")
    print(f"threshold used: {thr:.3f}; segments: {len(segs)}; wrote: {plays_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
