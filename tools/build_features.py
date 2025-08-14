#!/usr/bin/env python3
import argparse, json, sys, math, statistics
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple

# --- Utilities ---------------------------------------------------------------

def read_jsonl(p: Path) -> Iterable[Dict[str, Any]]:
    if not p.exists():
        return []
    with p.open() as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def safe_get(d: Dict[str, Any], key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

def bbox_area(b):
    # b = [x1,y1,x2,y2] or dict
    if isinstance(b, dict):
        x1,y1,x2,y2 = b.get("x1",0), b.get("y1",0), b.get("x2",0), b.get("y2",0)
    else:
        x1,y1,x2,y2 = b
    return max(0, x2-x1) * max(0, y2-y1)

def bbox_ar(b):
    if isinstance(b, dict):
        x1,y1,x2,y2 = b.get("x1",0), b.get("y1",0), b.get("x2",0), b.get("y2",0)
    else:
        x1,y1,x2,y2 = b
    w, h = max(1, x2-x1), max(1, y2-y1)
    return w / h

def robust_player_filter(boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Filter by confidence, min/max area, and plausible aspect ratio for a standing person
    out = []
    # approximate HD frame heuristic; adjust by frame area if available later
    MIN_AREA, MAX_AREA = 12*12, 200*200
    for b in boxes or []:
        conf = safe_get(b, "conf", 0.0) or safe_get(b, "score", 0.0) or 0.0
        if conf < 0.20:
            continue
        a = bbox_area(b)
        if a < MIN_AREA or a > MAX_AREA:
            continue
        ar = bbox_ar(b)
        if ar < 0.25 or ar > 1.2:
            continue
        out.append(b)
    return out

def motion_stats(tracks: List[Dict[str, Any]]) -> Tuple[float,float]:
    # Expect per-frame dx,dy or speed if present; otherwise derive from centers
    speeds = []
    for t in tracks:
        pts = safe_get(t, "points", []) or safe_get(t, "centers", [])
        for i in range(1, len(pts)):
            x0,y0 = pts[i-1][:2]
            x1,y1 = pts[i][:2]
            dx, dy = x1-x0, y1-y0
            speeds.append(math.hypot(dx, dy))
    if not speeds:
        return (0.0, 0.0)
    return (statistics.mean(speeds), statistics.median(speeds))

def summarize_segment(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    rows: all tracking rows for this seg_id (mixed frame data)
    Build schema-agnostic features usable by downstream classifier:
    - player_count_mean / max (after robust filtering)
    - bbox area stats
    - motion mean/median
    - track_count (unique ids)
    """
    boxes_per_frame = []
    all_boxes = []
    track_ids = set()
    for r in rows:
        # tolerate multiple naming styles
        boxes = safe_get(r, "boxes") or safe_get(r, "detections") or safe_get(r, "players") or []
        boxes = robust_player_filter(boxes)
        boxes_per_frame.append(len(boxes))
        all_boxes.extend(boxes)
        tid = safe_get(r, "track_id")
        if tid is not None:
            track_ids.add(tid)
        # Some trackers store per-frame track lists
        for tr in safe_get(r, "tracks", []):
            tid2 = safe_get(tr, "id")
            if tid2 is not None:
                track_ids.add(tid2)

    feats: Dict[str, Any] = {}
    feats["frames"] = len(rows)
    feats["player_count_mean"] = statistics.mean(boxes_per_frame) if boxes_per_frame else 0.0
    feats["player_count_max"]  = max(boxes_per_frame) if boxes_per_frame else 0
    feats["player_count_p50"]  = statistics.median(boxes_per_frame) if boxes_per_frame else 0.0
    feats["track_count"]       = len(track_ids)

    areas = [bbox_area(b) for b in all_boxes]
    if areas:
        feats["bbox_area_mean"] = statistics.mean(areas)
        feats["bbox_area_p50"]  = statistics.median(areas)
        feats["bbox_area_max"]  = max(areas)
    else:
        feats["bbox_area_mean"] = 0.0
        feats["bbox_area_p50"]  = 0.0
        feats["bbox_area_max"]  = 0.0

    # derive crude motion from any available tracks
    # collapse rows into pseudo-tracks if needed
    pseudo_tracks = []
    for r in rows:
        for tr in safe_get(r, "tracks", []):
            pseudo_tracks.append(tr)
    m_mean, m_med = motion_stats(pseudo_tracks)
    feats["motion_mean"] = m_mean
    feats["motion_p50"]  = m_med

    # “sufficient” heuristic: we want *some* signal, not 0 for everything
    feats["_sufficient"] = (feats["player_count_p50"] >= 8 and feats["player_count_max"] <= 30) or feats["track_count"] >= 10
    return feats

# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build per-segment features from tracking.jsonl in a schema-agnostic way.")
    ap.add_argument("--tracking", required=True, help="Path to tracking.jsonl")
    ap.add_argument("--segments", required=False, help="Path to plays.jsonl (for seg ordering & ids)")
    ap.add_argument("--out", required=True, help="Output features.jsonl path")
    args = ap.parse_args()

    tracking_p = Path(args.tracking)
    segments_p = Path(args.segments) if args.segments else None
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Load tracking rows and group by seg_id
    seg_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in read_jsonl(tracking_p):
        seg_id = safe_get(row, "seg_id") or safe_get(row, "segment") or safe_get(row, "id")
        if not seg_id:
            # tolerate tracking rows without seg_id; drop into special bucket
            seg_id = "__unknown__"
        seg_rows.setdefault(seg_id, []).append(row)

    # Determine ordered seg list
    ordered: List[str] = []
    if segments_p and segments_p.exists():
        for seg in read_jsonl(segments_p):
            sid = safe_get(seg, "id") or safe_get(seg, "seg_id")
            if sid:
                ordered.append(sid)
    else:
        ordered = list(seg_rows.keys())

    wrote = 0
    with out_p.open("w") as f:
        for sid in ordered:
            rows = seg_rows.get(sid, [])
            feats = summarize_segment(rows)
            rec = {"seg_id": sid, "features": feats}
            f.write(json.dumps(rec) + "\n")
            wrote += 1
            suff = "ok" if feats.get("_sufficient") else "weak"
            print(f"[feat] {sid}: players p50={feats['player_count_p50']:.1f} max={feats['player_count_max']} tracks={feats['track_count']} -> {suff}")

    print(f"[ok] features -> {str(out_p)}")
    print(f"[ok] feature rows: {wrote}")

if __name__ == "__main__":
    main()

