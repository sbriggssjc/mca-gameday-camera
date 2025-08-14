#!/usr/bin/env python3
"""
Quick detector sanity probe.

Usage:
  PYTHONPATH=. python3 tools/probe_detector.py --video video/manual_uploads/IMG_4129.MP4
Outputs:
  debug/first_frame.jpg
  debug/detector_probe.json
  debug/probe_overlay.jpg (if any detections found)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys, cv2

def draw_boxes(img, boxes):
    vis = img.copy()
    for b in boxes:
        cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0,255,0), 2)
        cv2.putText(vis, f'{b.get("label","obj")} {b.get("score",0):.2f}',
                    (b["x1"], max(0,b["y1"]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return vis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--max-frames", type=int, default=50)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from analysis.detectors import player_detector as det

    outdir = Path("debug"); outdir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    meta = {"opened": bool(ok), "shape": None, "frames_checked": 0, "total_detections": 0}
    if not ok:
        print("ERROR: could not open video:", args.video)
    else:
        meta["shape"] = list(frame.shape)
        cv2.imwrite(str(outdir/"first_frame.jpg"), frame)

        # warm up the background subtractor a bit
        for _ in range(15):
            ok2, f2 = cap.read()
            if not ok2: break
            det.player_detector(f2)

        # analyze frames
        det_frame = None
        for _ in range(args.max_frames):
            ok3, f3 = cap.read()
            if not ok3: break
            boxes = det.player_detector(f3)
            meta["total_detections"] += len(boxes)
            if det_frame is None and boxes:
                det_frame = (f3, boxes)
            meta["frames_checked"] += 1

        if det_frame is not None:
            vis = draw_boxes(det_frame[0], det_frame[1])
            cv2.imwrite(str(outdir/"probe_overlay.jpg"), vis)

    (outdir/"detector_probe.json").write_text(json.dumps(meta, indent=2))
    print("Wrote:", outdir/"first_frame.jpg", "and", outdir/"detector_probe.json")
    if (outdir/"probe_overlay.jpg").exists():
        print("Wrote:", outdir/"probe_overlay.jpg")

if __name__ == "__main__":
    main()
