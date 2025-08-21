#!/usr/bin/env python3
import argparse, json, math, statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, DefaultDict
from collections import defaultdict
from tools.json_io import iter_jsonl_safe

def read_jsonl(p: Path) -> Iterable[Dict[str, Any]]:
    return iter_jsonl_safe(p)

def safe_get(d: Dict[str, Any], key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

def bbox_from_any(b: Any) -> Optional[Tuple[float,float,float,float]]:
    if b is None: return None
    if isinstance(b, dict):
        x1,y1,x2,y2 = b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2")
        if None in (x1,y1,x2,y2): return None
        return float(x1),float(y1),float(x2),float(y2)
    if isinstance(b, (list,tuple)) and len(b)>=4:
        return float(b[0]),float(b[1]),float(b[2]),float(b[3])
    return None

def bbox_area(bb: Tuple[float,float,float,float]) -> float:
    x1,y1,x2,y2 = bb
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def bbox_ar(bb: Tuple[float,float,float,float]) -> float:
    x1,y1,x2,y2 = bb
    w,h = max(1.0, x2-x1), max(1.0, y2-y1)
    return w/h

def robust_player_filter(bboxes: List[Tuple[float,float,float,float]], confs: Optional[List[float]]=None) -> List[Tuple[float,float,float,float]]:
    out = []
    MIN_A, MAX_A = 12*12, 200*200
    for i,bb in enumerate(bboxes):
        conf = (confs[i] if confs and i < len(confs) else 1.0) or 0.0
        if conf < 0.20: continue
        a = bbox_area(bb)
        if a < MIN_A or a > MAX_A: continue
        ar = bbox_ar(bb)
        if ar < 0.25 or ar > 1.2: continue
        out.append(bb)
    return out

def build_seg_index(plays_path: Path):
    """Return (ordered_ids, seg_range_by_id) with frame ranges if present."""
    ordered_ids: List[str] = []
    seg_ranges: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    for seg in read_jsonl(plays_path):
        sid = safe_get(seg, "id") or safe_get(seg, "seg_id")
        if not sid: continue
        ordered_ids.append(sid)

        start = (safe_get(seg, "start_frame", None)
                 if "start_frame" in seg else
                 safe_get(seg, "f0", None) if "f0" in seg else
                 safe_get(seg, "start_idx", None))
        end   = (safe_get(seg, "end_frame", None)
                 if "end_frame" in seg else
                 safe_get(seg, "f1", None) if "f1" in seg else
                 safe_get(seg, "end_idx", None))
        start = int(start) if isinstance(start, (int,float)) else None
        end   = int(end)   if isinstance(end,   (int,float)) else None
        seg_ranges[sid] = (start, end)
    return ordered_ids, seg_ranges

def which_seg_for_frame(frame: int, seg_ranges: Dict[str, Tuple[Optional[int], Optional[int]]]) -> Optional[str]:
    for sid,(s,e) in seg_ranges.items():
        if s is not None and e is not None and s <= frame <= e:
            return sid
    return None

def summarize_segment(frames_to_boxes: Dict[int, List[Tuple[float,float,float,float]]]) -> Dict[str, Any]:
    counts = [len(bs) for _,bs in frames_to_boxes.items()]
    feats: Dict[str, Any] = {}
    feats["frames"] = len(frames_to_boxes)
    feats["player_count_mean"] = statistics.mean(counts) if counts else 0.0
    feats["player_count_p50"]  = statistics.median(counts) if counts else 0.0
    feats["player_count_max"]  = max(counts) if counts else 0
    areas: List[float] = [bbox_area(bb) for bbs in frames_to_boxes.values() for bb in bbs]
    feats["bbox_area_mean"] = (statistics.mean(areas) if areas else 0.0)
    feats["bbox_area_p50"]  = (statistics.median(areas) if areas else 0.0)
    feats["bbox_area_max"]  = (max(areas) if areas else 0.0)
    feats["_sufficient"] = (feats["player_count_p50"] >= 8 and feats["player_count_max"] <= 30)
    return feats

def main():
    ap = argparse.ArgumentParser(description="Schema-agnostic feature builder.")
    ap.add_argument("--tracking", required=True, help="tracking.jsonl path")
    ap.add_argument("--segments", required=False, help="plays.jsonl (ordering / frame mapping)")
    ap.add_argument("--out", required=True, help="features.jsonl output path")
    args = ap.parse_args()

    tracking_p = Path(args.tracking)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ordered: List[str] = []
    seg_ranges: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    if args.segments:
        segs_p = Path(args.segments)
        if segs_p.exists():
            ordered, seg_ranges = build_seg_index(segs_p)

    # seg_id -> frame -> [bboxes]
    buckets: DefaultDict[str, DefaultDict[int, List[Tuple[float,float,float,float]]]] = defaultdict(lambda: defaultdict(list))

    for row in read_jsonl(tracking_p):
        sid = safe_get(row, "seg_id") or safe_get(row, "segment") or None

        bboxes: List[Tuple[float,float,float,float]] = []
        confs: List[float] = []

        # Per-frame list of boxes
        if isinstance(safe_get(row, "boxes"), list):
            for b in safe_get(row, "boxes"):
                bb = bbox_from_any(b)
                if bb:
                    bboxes.append(bb)
                    confs.append((safe_get(b,"conf") or safe_get(b,"score") or 1.0))
        elif isinstance(safe_get(row, "detections"), list):
            for b in safe_get(row, "detections"):
                bb = bbox_from_any(b)
                if bb:
                    bboxes.append(bb)
                    confs.append((safe_get(b,"conf") or safe_get(b,"score") or 1.0))
        else:
            # Per-player jersey rows with single bbox
            bb = bbox_from_any(safe_get(row, "bbox"))
            if bb:
                bboxes.append(bb)
                confs.append(1.0)

        if not bboxes and "frame" not in row and not sid:
            continue

        bboxes = robust_player_filter(bboxes, confs if confs else None)

        frame = safe_get(row, "frame", 0)
        try:
            frame = int(frame)
        except Exception:
            frame = 0

        if not sid and seg_ranges:
            sid = which_seg_for_frame(frame, seg_ranges)
        if not sid:
            sid = "__unknown__"

        for bb in bboxes:
            buckets[sid][frame].append(bb)

    seg_ids = (ordered if ordered else list(buckets.keys()))

    wrote = 0
    with out_p.open("w") as f:
        for sid in seg_ids:
            feats = summarize_segment(buckets.get(sid, {}))
            rec = {"seg_id": sid, "features": feats}
            f.write(json.dumps(rec)+"\n")
            wrote += 1
            suff = "ok" if feats.get("_sufficient") else "weak"
            print(f"[feat] {sid}: players p50={feats['player_count_p50']:.1f} max={feats['player_count_max']} frames={feats['frames']} -> {suff}")

    print(f"[ok] features -> {str(out_p)}")
    print(f"[ok] feature rows: {wrote}")

if __name__ == "__main__":
    main()

