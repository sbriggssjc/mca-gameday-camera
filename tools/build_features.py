#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import cv2

# Import the detector exactly as pipeline does
from analysis.detectors import player_detector as det

def sample_frames(cap, seg, stride=3, max_frames=60):
    """Yield (frame_idx, frame_bgr) for a segment [start,end) in seconds."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start = int(seg["start"] * fps)
    end   = int(seg["end"]   * fps)
    if end <= start: return
    count = 0
    for i in range(start, end, stride):
        if count >= max_frames: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok: break
        yield i, frame
        count += 1

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--plays",  required=True, help="output/.../plays.jsonl from pipeline")
    ap.add_argument("--outdir", required=True, help="same output dir (e.g., output/IMG_4129_... )")
    ap.add_argument("--stride", type=int, default=3, help="frame sampling stride")
    ap.add_argument("--max-per-seg", type=int, default=60, help="max frames per segment to process")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    # Load segments the pipeline already made
    segs = [json.loads(l) for l in open(args.plays)]
    track_rows = []
    feat_rows  = []

    # One detector instance (cheaper on Jetson)
    detector = det.PlayerDetector()

    for seg in segs:
        sid   = seg["id"]
        n_det = 0
        # Per-segment aggregation
        total_boxes, total_area = 0, 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        frame_area = float(w*h)

        for frame_idx, frame in sample_frames(cap, seg, stride=args.stride, max_frames=args.max_per_seg):
            boxes = detector.detect(frame)  # list of dicts with x1,y1,x2,y2,score,label
            if boxes:
                for b in boxes:
                    track_rows.append({
                        "segment_id": sid,
                        "frame_index": frame_idx,
                        "bbox": [int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])],
                        "score": float(b.get("score", 1.0)),
                        "label": b.get("label", "player")
                    })
                    total_boxes += 1
                    total_area  += max(0, (b["x2"]-b["x1"])) * max(0, (b["y2"]-b["y1"]))
                n_det += len(boxes)

        # Minimal, generic features per segment that many pipelines can consume
        feat_rows.append({
            "segment_id": sid,
            "duration": seg.get("end", 0) - seg.get("start", 0),
            "num_frames_scanned": min(args.max_per_seg, max(0, int((seg.get("end",0)-seg.get("start",0))* (cap.get(cv2.CAP_PROP_FPS) or 30.0)/args.stride))),
            "num_detections": int(total_boxes),
            "avg_bbox_fill": float(total_area/frame_area) if frame_area > 0 else 0.0,
            "has_players": bool(total_boxes > 0)
        })

        print(f"[features] seg {sid}: detections={n_det}, total_boxes={total_boxes}")

    # Write outputs where the pipeline looked earlier
    outdir = Path(args.outdir)
    write_jsonl(outdir/"tracking.jsonl", track_rows)
    write_jsonl(outdir/"features.jsonl", feat_rows)

    # Also write a tiny summary for sanity
    summary = {
        "segments": len(segs),
        "tracking_rows": len(track_rows),
        "feature_rows": len(feat_rows),
        "nonempty_segments": sum(1 for r in feat_rows if r["has_players"])
    }
    with open(outdir/"features_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[features] wrote:",
          outdir/"tracking.jsonl",
          outdir/"features.jsonl",
          outdir/"features_summary.json")
if __name__ == "__main__":
    main()
