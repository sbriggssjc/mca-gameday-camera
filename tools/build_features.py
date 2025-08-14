#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, Tuple, List
import cv2

try:
    from analysis.detectors import player_detector as det
except Exception as e:
    print(f"[ERR] Could not import analysis.detectors.player_detector: {e}", file=sys.stderr)
    sys.exit(2)

SEG_ID_KEYS = ("id", "sid", "seg_id", "name", "idx")
START_KEYS  = ("start", "t0", "start_s", "begin", "ts")
END_KEYS    = ("end", "t1", "end_s", "stop", "finish", "te")

def first_key(d: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in d: return k
    return None

def extract_seg_fields(seg: Dict[str, Any], fallback_sid: str) -> Optional[Tuple[str, float, float]]:
    sid_key = first_key(seg, SEG_ID_KEYS)
    sid = str(seg[sid_key]) if sid_key else fallback_sid
    t0_key = first_key(seg, START_KEYS)
    t1_key = first_key(seg, END_KEYS)
    if t0_key and t1_key:
        try:
            t0, t1 = float(seg[t0_key]), float(seg[t1_key])
            if t1 > t0 >= 0.0: return sid, t0, t1
        except Exception: return None
    if "ts" in seg and "duration" in seg:
        try:
            t0 = float(seg["ts"]); t1 = t0 + float(seg["duration"])
            if t1 > t0 >= 0.0: return sid, t0, t1
        except Exception: return None
    return None

def frames_for_interval(cap, t0, t1, stride, limit):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    a = max(0, int(round(t0 * fps))); b = max(a, int(round(t1 * fps)) - 1)
    idxs = list(range(a, b + 1, max(1, stride)))
    if limit is not None and len(idxs) > limit: idxs = idxs[:limit]
    return [(i, i / fps) for i in idxs]

def run_detector_on_frame(frame):
    if callable(getattr(det, "player_detector", None)):
        boxes = det.player_detector(frame)
    elif hasattr(det, "Detector"):
        _inst = getattr(det, "_GLOBAL_DET", None)
        if _inst is None:
            _inst = det.Detector(); setattr(det, "_GLOBAL_DET", _inst)
        boxes = _inst.detect(frame)
    else:
        raise RuntimeError("player_detector module has no callable entrypoint")
    return list(boxes or [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--plays", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-per-seg", type=int, default=48)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    track_p = outdir / "tracking.jsonl"; feat_p = outdir / "features.jsonl"

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] Failed to open video: {args.video}", file=sys.stderr); sys.exit(2)

    segs = []
    with open(args.plays) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try: seg = json.loads(line)
            except Exception as e:
                if args.verbose: print(f"[WARN] bad JSON line {i}: {e}", file=sys.stderr)
                continue
            meta = extract_seg_fields(seg, fallback_sid=f"seg_{i-1:04d}")
            if meta is None:
                if args.verbose: print(f"[WARN] skip seg {i-1}: no start/end", file=sys.stderr)
                continue
            sid, t0, t1 = meta
            segs.append((sid, t0, t1))
    if not segs:
        print("[ERR] No usable segments in plays.jsonl", file=sys.stderr); sys.exit(3)

    n_track_rows = 0
    with open(track_p, "w") as tf, open(feat_p, "w") as ff:
        for sid, t0, t1 in segs:
            samples = frames_for_interval(cap, t0, t1, args.stride, args.max_per_seg)
            counts, areas_all, written = [], [], 0
            for frame_idx, ts in samples:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    if args.verbose: print(f"[WARN] unreadable frame {frame_idx} @ {ts:.3f}s", file=sys.stderr)
                    continue
                boxes = run_detector_on_frame(frame)
                areas = []
                for b in boxes:
                    try:
                        w = max(0, int(b["x2"]) - int(b["x1"]))
                        h = max(0, int(b["y2"]) - int(b["y1"]))
                        areas.append(float(w*h))
                    except Exception:
                        pass
                areas_all.extend(areas); counts.append(len(boxes))
                tf.write(json.dumps({"seg": sid, "t": ts, "frame": frame_idx, "boxes": boxes, "n": len(boxes)}) + "\n")
                n_track_rows += 1; written += 1

            nframes = len(counts)
            avg_players = (sum(counts)/nframes) if nframes else 0.0
            max_players = max(counts) if counts else 0
            ff.write(json.dumps({
                "seg": sid, "t0": t0, "t1": t1, "nframes": nframes,
                "avg_players": avg_players, "max_players": max_players,
                "has_players": bool(max_players > 0),
                "avg_box_area": (sum(areas_all)/len(areas_all)) if areas_all else 0.0,
                "players_count_mean": avg_players, "players_count_max": max_players
            }) + "\n")
            if args.verbose:
                print(f"[seg {sid}] frames={written} avg_players={avg_players:.2f} max={max_players}")

    cap.release()
    print(f"[ok] tracking -> {track_p}")
    print(f"[ok] features -> {feat_p}")
    print(f"[ok] tracking rows: {n_track_rows}, segments written: {len(segs)}")

if __name__ == "__main__":
    main()
