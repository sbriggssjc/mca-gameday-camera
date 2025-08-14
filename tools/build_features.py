#!/usr/bin/env python3
"""
Schema-agnostic feature builder.

Reads segments from plays.jsonl (with flexible field names), samples frames from
the source video, runs the pipeline's player detector, and writes:

  <outdir>/tracking.jsonl  # per-frame detections
  <outdir>/features.jsonl  # per-segment rollups

Boxes format:
  {"x1": int, "y1": int, "x2": int, "y2": int, "score": float, "label": "player"}

Usage:
  PYTHONPATH=. python3 tools/build_features.py \
    --video video/manual_uploads/IMG_4129.MP4 \
    --plays output/IMG_4129_YYYYMMDD_HHMM/plays.jsonl \
    --outdir output/IMG_4129_YYYYMMDD_HHMM \
    --stride 4 \
    --max-per-seg 48
"""

import argparse, json, sys
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, Tuple, List

import cv2

# Import the detector exactly as analysis.pipeline expects
try:
    from analysis.detectors import player_detector as det
except Exception as e:
    print(f"[ERR] Could not import analysis.detectors.player_detector: {e}", file=sys.stderr)
    sys.exit(2)

# ----------------------- helpers -----------------------

SEG_ID_KEYS = ("id", "sid", "seg_id", "name", "idx")
START_KEYS  = ("start", "t0", "start_s", "begin", "ts")
END_KEYS    = ("end", "t1", "end_s", "stop", "finish", "te")

def first_key(d: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None

def extract_seg_fields(seg: Dict[str, Any], fallback_sid: str) -> Optional[Tuple[str, float, float]]:
    """
    Returns (sid, t0, t1) in seconds, or None if times cannot be resolved.
    Supports (start,end) or (ts,duration).
    """
    sid_key = first_key(seg, SEG_ID_KEYS)
    sid = str(seg[sid_key]) if sid_key else fallback_sid

    t0_key = first_key(seg, START_KEYS)
    t1_key = first_key(seg, END_KEYS)

    if t0_key and t1_key:
        try:
            t0 = float(seg[t0_key])
            t1 = float(seg[t1_key])
            if t1 > t0 >= 0.0:
                return sid, t0, t1
        except Exception:
            return None

    # Fallback: ts + duration
    if "ts" in seg and "duration" in seg:
        try:
            t0 = float(seg["ts"])
            t1 = t0 + float(seg["duration"])
            if t1 > t0 >= 0.0:
                return sid, t0, t1
        except Exception:
            return None

    return None

def frames_for_interval(cap: cv2.VideoCapture, t0: float, t1: float, stride: int, limit: Optional[int]) -> List[Tuple[int, float]]:
    """
    Returns a list of (frame_idx, time_s) to sample between [t0, t1].
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_idx = max(0, int(round(t0 * fps)))
    end_idx   = max(start_idx, int(round(t1 * fps)) - 1)
    idxs = list(range(start_idx, end_idx + 1, max(1, stride)))
    if limit is not None and len(idxs) > limit:
        idxs = idxs[:limit]
    return [(i, i / fps) for i in idxs]

def run_detector_on_frame(frame) -> List[Dict[str, Any]]:
    """
    Calls either module-level function or class API, depending on what's available.
    """
    if callable(getattr(det, "player_detector", None)):
        boxes = det.player_detector(frame)
    elif hasattr(det, "Detector"):  # compat with alt detectors
        _inst = getattr(det, "_GLOBAL_DET", None)
        if _inst is None:
            _inst = det.Detector()  # type: ignore[attr-defined]
            setattr(det, "_GLOBAL_DET", _inst)
        boxes = _inst.detect(frame)  # type: ignore[assignment]
    else:
        raise RuntimeError("player_detector module has no callable entrypoint")

    if boxes is None:
        return []
    return list(boxes)

# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to source video (same one used by pipeline)")
    ap.add_argument("--plays", required=True, help="plays.jsonl produced by analysis.pipeline")
    ap.add_argument("--outdir", required=True, help="output directory (same folder as plays.jsonl)")
    ap.add_argument("--stride", type=int, default=4, help="sample every Nth frame in each segment")
    ap.add_argument("--max-per-seg", type=int, default=48, help="cap frames sampled per segment")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tracking_path = outdir / "tracking.jsonl"
    features_path = outdir / "features.jsonl"

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] Failed to open video: {args.video}", file=sys.stderr)
        sys.exit(2)

    # Load segments
    segs: List[Dict[str, Any]] = []
    with open(args.plays, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                seg = json.loads(line)
            except Exception as e:
                if args.verbose:
                    print(f"[WARN] Skipping bad JSON on line {i+1}: {e}", file=sys.stderr)
                continue

            meta = extract_seg_fields(seg, fallback_sid=f"seg_{i:04d}")
            if meta is None:
                if args.verbose:
                    print(f"[WARN] Skipping segment {i}: could not resolve start/end seconds", file=sys.stderr)
                continue
            sid, t0, t1 = meta
            segs.append({"_sid": sid, "_t0": t0, "_t1": t1})

    if not segs:
        print("[ERR] No usable segments found in plays.jsonl (missing start/end?)", file=sys.stderr)
        sys.exit(3)

    # Process segments
    n_track_rows = 0
    with open(tracking_path, "w") as track_f, open(features_path, "w") as feat_f:
        for seg in segs:
            sid, t0, t1 = seg["_sid"], seg["_t0"], seg["_t1"]
            samples = frames_for_interval(cap, t0, t1, stride=args.stride, limit=args.max_per_seg)

            counts: List[int] = []
            areas_all: List[float] = []
            written_frames = 0

            for frame_idx, ts in samples:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    if args.verbose:
                        print(f"[WARN] Unable to read frame {frame_idx} (ts={ts:.3f}s)")
                    continue

                boxes = run_detector_on_frame(frame)

                # compute simple areas for rollups
                areas = []
                for b in boxes:
                    try:
                        w = max(0, int(b["x2"]) - int(b["x1"]))
                        h = max(0, int(b["y2"]) - int(b["y1"]))
                        areas.append(float(w * h))
                    except Exception:
                        pass
                areas_all.extend(areas)
                counts.append(len(boxes))

                # write tracking row
                row = {
                    "seg": sid,
                    "t": float(ts),
                    "frame": int(frame_idx),
                    "boxes": boxes,
                    "n": int(len(boxes)),
                }
                track_f.write(json.dumps(row) + "\n")
                n_track_rows += 1
                written_frames += 1

            # per-segment features
            if counts:
                nframes = len(counts)
                avg_players = sum(counts) / nframes
                max_players = max(counts)
            else:
                nframes = 0
                avg_players = 0.0
                max_players = 0

            feat_row = {
                "seg": sid,
                "t0": t0,
                "t1": t1,
                "nframes": nframes,
                "avg_players": avg_players,
                "max_players": max_players,
                "has_players": bool(max_players > 0),
                "avg_box_area": (sum(areas_all) / len(areas_all)) if areas_all else 0.0,
                # common aliases some codepaths check for
                "players_count_mean": avg_players,
                "players_count_max": max_players,
            }
            feat_f.write(json.dumps(feat_row) + "\n")

            if args.verbose:
                print(f"[seg {sid}] frames={written_frames} avg_players={avg_players:.2f} max={max_players}")

    cap.release()
    print(f"[ok] tracking -> {tracking_path}")
    print(f"[ok] features -> {features_path}")
    print(f"[ok] tracking rows: {n_track_rows}, segments written: {len(segs)}")

if __name__ == "__main__":
    main()
