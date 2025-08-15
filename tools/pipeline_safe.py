#!/usr/bin/env python3
import argparse, json, subprocess, sys
from pathlib import Path
import cv2

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r)+"\n")

def fallback_segments(video: str, outdir: Path, min_len_s: float, gap_s: float):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (nframes / fps) if nframes else 0
    if duration <= 0:
        # Try to read a handful of frames to estimate duration if frame count is missing
        cnt = 0
        while cnt < 300:
            ok, _ = cap.read()
            if not ok: break
            cnt += 1
        duration = max(duration, cnt / fps)
    cap.release()

    segs = []
    t = 0.0
    idx = 0
    while t + min_len_s <= duration + 1e-6:
        segs.append({
            "play_id": idx+1,
            "start_s": round(t, 3),
            "end_s": round(min(t + min_len_s, duration), 3),
            "source": "fallback_windowize",
            "segment_id": f"seg_{idx:04d}",
        })
        idx += 1
        t += (min_len_s + gap_s)
    write_jsonl(outdir / "plays.jsonl", segs)
    return len(segs), fps

def run(cmd):
    print(">>", " ".join(cmd))
    return subprocess.run(cmd, check=False)

def main():
    ap = argparse.ArgumentParser(description="Safe fallback pipeline: segments -> detector -> features")
    ap.add_argument("--video", required=True)
    ap.add_argument("--team", default="WHITE")
    ap.add_argument("--playbook", default="mca_full_playbook_final.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-play-length", type=float, default=6.0)
    ap.add_argument("--min-play-gap", type=float, default=1.5)
    ap.add_argument("--detect-model", default="player_detector")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) fallback segmentation
    nsegs, fps = fallback_segments(args.video, outdir, args.min_play_length, args.min_play_gap)
    print(f"[segmenter] Segments written: {nsegs} (fallback={nsegs}) -> {outdir/'plays.jsonl'}  | fps~{fps:.2f}")

    # 2) detector -> tracking
    print("[detect] running tools/detect_to_tracking.py …")
    rc1 = run([sys.executable, "tools/detect_to_tracking.py",
               "--video", args.video,
               "--plays", str(outdir/"plays.jsonl"),
               "--out", str(outdir/"tracking.jsonl"),
               "--model", args.detect_model,
               "--stride", str(args.stride)])
    if rc1.returncode != 0:
        print("[detect] WARN: detector step returned non-zero")

    # 3) features
    print("[feat] running tools/build_features.py …")
    rc2 = run([sys.executable, "tools/build_features.py",
               "--tracking", str(outdir/"tracking.jsonl"),
               "--segments", str(outdir/"plays.jsonl"),
               "--out", str(outdir/"features.jsonl")])
    if rc2.returncode != 0:
        print("[feat] WARN: feature build returned non-zero")

    # 4) quick summary
    feats = list()
    try:
        with (outdir/"features.jsonl").open() as f:
            for line in f:
                if line.strip():
                    feats.append(json.loads(line))
    except Exception:
        pass
    ok = sum(1 for r in feats if r.get("features",{}).get("_sufficient"))
    print("\n==== Debug Summary (safe pipeline) ====")
    print(f"Output dir: {outdir}")
    print(f"Segments: {nsegs} | Features rows: {len(feats)} | OK: {ok} | WEAK: {len(feats)-ok}")
    print("=======================================\n")

if __name__ == "__main__":
    main()
