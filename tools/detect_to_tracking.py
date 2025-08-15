#!/usr/bin/env python3
import argparse, json, importlib
from pathlib import Path
import cv2

def read_jsonl(p: Path):
    with p.open() as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def plays_from_jsonl(p: Path):
    plays=[]
    for seg in read_jsonl(p):
        sid = seg.get("segment_id") or seg.get("seg_id") or seg.get("id")
        # support fallback windowizer schema with seconds
        s  = seg.get("start_frame", seg.get("f0", seg.get("start_idx", seg.get("start_s", seg.get("start")))))
        e  = seg.get("end_frame",   seg.get("f1", seg.get("end_idx", seg.get("end_s",   seg.get("end")))))
        plays.append({"sid": sid, "start": s, "end": e})
    return plays

def load_detector(model_name: str):
    mod = importlib.import_module(f"analysis.detectors.{model_name}")
    for name in ("Detector","PlayerDetector","Model","ModelDetector"):
        cls = getattr(mod, name, None)
        if cls: return cls()
    # fallback: module exposes a factory
    for name in ("load","create","get"):
        fn = getattr(mod, name, None)
        if callable(fn): return fn()
    raise RuntimeError(f"Could not construct detector from analysis.detectors.{model_name}")

def normalize_boxes(items):
    """Return a list of dicts with x1,y1,x2,y2,score keys."""
    out=[]
    for it in (items or []):
        if isinstance(it, dict):
            if all(k in it for k in ("x1","y1","x2","y2")):
                out.append({"x1":float(it["x1"]),"y1":float(it["y1"]),
                            "x2":float(it["x2"]),"y2":float(it["y2"]),
                            "score":float(it.get("conf", it.get("score",1.0)))})
                continue
            if "bbox" in it and isinstance(it["bbox"], (list,tuple)) and len(it["bbox"])>=4:
                x1,y1,x2,y2 = it["bbox"][:4]
                out.append({"x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2),
                            "score":float(it.get("conf", it.get("score",1.0)))})
                continue
        elif isinstance(it, (list,tuple)) and len(it)>=4:
            # [x1,y1,x2,y2,(score)]
            x1,y1,x2,y2 = it[:4]
            score = float(it[4]) if len(it)>4 else 1.0
            out.append({"x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2),"score":score})
    return out

def main():
    ap = argparse.ArgumentParser(description="Run a detector over plays.jsonl windows, write tracking.jsonl with per-frame boxes.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--plays", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="player_detector")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    args = ap.parse_args()

    plays_p = Path(args.plays)
    out_p   = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-6: fps = 30.0
    detector = load_detector(args.model)

    # Build segment windows (support seconds or frames)
    segs = []
    for rec in plays_from_jsonl(plays_p):
        sid, s, e = rec["sid"], rec["start"], rec["end"]
        if sid is None: continue
        if s is None or e is None:
            continue
        if isinstance(s, float) or isinstance(e, float):
            # seconds → frames
            s = int(round(float(s)*fps))
            e = int(round(float(e)*fps))
        s = int(s); e = int(e)
        if e < s: s,e = e,s
        segs.append((sid, s, e))

    wrote=0
    with out_p.open("w") as fout:
        for sid, f0, f1 in segs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            f = f0
            while f <= f1:
                ok, frame = cap.read()
                if not ok: break
                if (f - f0) % args.stride != 0:
                    f += 1
                    continue
                detections = detector.detect(frame)
                boxes = normalize_boxes(detections)
                rec = {"seg_id": sid, "frame": int(f), "boxes": boxes}
                fout.write(json.dumps(rec) + "\n")
                wrote += 1
                f += 1
    cap.release()
    print(f"[ok] tracking -> {str(out_p)} rows={wrote}, stride={args.stride}, fps~{fps:.2f}")

if __name__ == "__main__":
    main()
