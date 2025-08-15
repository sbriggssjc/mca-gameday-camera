#!/usr/bin/env python3
import argparse, json, statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, DefaultDict
from collections import defaultdict

def read_jsonl(p: Path) -> Iterable[Dict[str, Any]]:
    if not p.exists(): return []
    with p.open() as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: yield json.loads(line)
            except Exception: continue

def g(d,k,default=None): return d.get(k,default) if isinstance(d,dict) else default

def bbox_from_any(b: Any):
    if b is None: return None
    if isinstance(b, dict):
        x1,y1,x2,y2 = b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2")
        if None in (x1,y1,x2,y2): return None
        return float(x1),float(y1),float(x2),float(y2)
    if isinstance(b,(list,tuple)) and len(b)>=4:
        return float(b[0]),float(b[1]),float(b[2]),float(b[3])
    return None

def area(bb): x1,y1,x2,y2 = bb; return max(0.0,x2-x1)*max(0.0,y2-y1)
def ar(bb):
    x1,y1,x2,y2 = bb; w,h = max(1.0,x2-x1), max(1.0,y2-y1); return w/h

def filter_players(bbs, confs=None):
    out=[]
    # Easier thresholds (your sample had 10x10 bboxes = 100 px^2)
    MIN_A, MAX_A = 8*8, 400*400
    MIN_CONF = 0.10
    for i,bb in enumerate(bbs):
        conf=(confs[i] if confs and i<len(confs) else 1.0) or 0.0
        if conf<MIN_CONF: continue
        a=area(bb)
        if a<MIN_A or a>MAX_A: continue
        r=ar(bb)
        if r<0.20 or r>1.6: continue
        out.append(bb)
    return out

def seg_index(plays_p: Path):
    """
    Build ordered ids + (optional) frame ranges.
    Accept id keys: id, seg_id, segment_id, name; if missing, auto-assign seg_{i:04d}.
    Accept range keys: start_frame/f0/start_idx/start and end_frame/f1/end_idx/end.
    """
    ordered=[]; ranges={}
    for i, seg in enumerate(read_jsonl(plays_p)):
        sid = (g(seg,"id") or g(seg,"seg_id") or g(seg,"segment_id") or
               g(seg,"name") or f"seg_{i:04d}")
        ordered.append(sid)
        s = g(seg,"start_frame", g(seg,"f0", g(seg,"start_idx", g(seg,"start"))))
        e = g(seg,"end_frame",   g(seg,"f1", g(seg,"end_idx", g(seg,"end"))))
        s = int(s) if isinstance(s,(int,float)) else None
        e = int(e) if isinstance(e,(int,float)) else None
        ranges[sid]=(s,e)
    return ordered, ranges

def seg_for_frame(frame: int, ranges):
    for sid,(s,e) in ranges.items():
        if s is not None and e is not None and s<=frame<=e: return sid
    return None

def summarize(frames_to_boxes):
    counts=[len(bs) for _,bs in frames_to_boxes.items()]
    feats={"frames": len(frames_to_boxes)}
    feats["player_count_mean"]=statistics.mean(counts) if counts else 0.0
    feats["player_count_p50"]=statistics.median(counts) if counts else 0.0
    feats["player_count_max"]=max(counts) if counts else 0
    areas=[area(bb) for bbs in frames_to_boxes.values() for bb in bbs]
    feats["bbox_area_mean"]=statistics.mean(areas) if areas else 0.0
    feats["bbox_area_p50"]=statistics.median(areas) if areas else 0.0
    feats["bbox_area_max"]=max(areas) if areas else 0.0
    feats["_sufficient"] = (feats["player_count_p50"] >= 6 and feats["player_count_max"]<=30)
    return feats

def main():
    ap=argparse.ArgumentParser(description="Schema-agnostic feature builder.")
    ap.add_argument("--tracking", required=True)
    ap.add_argument("--segments", required=False)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    tracking_p=Path(args.tracking); out_p=Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)

    ordered=[]; ranges={}
    if args.segments and Path(args.segments).exists():
        ordered, ranges = seg_index(Path(args.segments))

    buckets: DefaultDict[str, DefaultDict[int, List[Tuple[float,float,float,float]]]] = defaultdict(lambda: defaultdict(list))

    for row in read_jsonl(tracking_p):
        sid = g(row,"seg_id") or g(row,"segment")
        bbs=[]; confs=[]
        if isinstance(g(row,"boxes"), list):
            for b in g(row,"boxes"):
                bb=bbox_from_any(b)
                if bb: bbs.append(bb); confs.append(g(b,"conf", g(b,"score",1.0)))
        elif isinstance(g(row,"detections"), list):
            for b in g(row,"detections"):
                bb=bbox_from_any(b)
                if bb: bbs.append(bb); confs.append(g(b,"conf", g(b,"score",1.0)))
        else:
            bb=bbox_from_any(g(row,"bbox"))
            if bb: bbs.append(bb); confs.append(1.0)

        bbs=filter_players(bbs, confs if confs else None)
        # If we still have nothing and no frame/seg_id, skip
        if not bbs and ("frame" not in row) and not sid: continue

        try: frame=int(g(row,"frame",0))
        except Exception: frame=0

        if not sid and ranges: sid=seg_for_frame(frame, ranges)
        if not sid: sid="__unknown__"

        for bb in bbs: buckets[sid][frame].append(bb)

    # Guarantee we write one row per segment in plays.jsonl (even if tracking is empty)
    seg_ids = (ordered if ordered else list(buckets.keys())) or ["__unknown__"]

    wrote=0
    with out_p.open("w") as f:
        for sid in seg_ids:
            feats=summarize(buckets.get(sid, {}))
            f.write(json.dumps({"seg_id":sid,"features":feats})+"\n")
            wrote+=1
            flag="ok" if feats.get("_sufficient") else "weak"
            print(f"[feat] {sid}: players p50={feats['player_count_p50']:.1f} max={feats['player_count_max']} frames={feats['frames']} -> {flag}")
    print(f"[ok] features -> {out_p}")
    print(f"[ok] feature rows: {wrote}")

if __name__=="__main__": main()
