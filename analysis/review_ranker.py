import json, math
from pathlib import Path
from typing import Dict, Any, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def _score_clip(play: Dict[str, Any], feat: Dict[str, Any], pb) -> Dict[str, Any]:
    """
    Heuristic scoring:
      +3 misalignment to playbook (formation/family mismatch)
      +2 contain/edge fail (monster/blood), +2 unmanned gap vs base fits
      +2 explosive allowed or TFL against us, +1 negative outcome
      +1 player_count anomaly or spacing anomaly
    """
    score, reasons = 0.0, []
    # formation/family mismatch (requires that upstream attached intended 'play_name' or 'family')
    called = (play.get("called_play") or play.get("play_name") or "").lower()
    off = pb.get_offense_play_by_name(called) if called else None
    if off:
        detected_family = (feat.get("offense_family") or "").lower()
        if detected_family and detected_family != off["family"].lower():
            score += 3
            reasons.append(f"Mismatch: expected family={off['family']} got {detected_family}")
        detected_form = (feat.get("formation") or "").title()
        if detected_form and detected_form != off["formation"]:
            score += 2
            reasons.append(f"Formation mismatch: expected {off['formation']} got {detected_form}")
    # contain / edge failure
    if feat.get("edge_broken_right") or feat.get("edge_broken_left"):
        score += 2
        reasons.append("Edge/contain lost (Monster/Blood)")
    # unmanned gap (simple proxy: run_hit_gap not in {A,B,C,D} responsibility seen)
    hit_gap = (feat.get("run_hit_gap") or "").upper()
    if hit_gap in {"A", "B", "C", "D"} and feat.get("gap_unfilled", False):
        score += 2
        reasons.append(f"Unfilled gap: {hit_gap}")
    # outcome
    if feat.get("explosive_gain_allowed") or feat.get("explosive_gain_for"):
        score += 2
        reasons.append("Explosive play")
    if feat.get("tfl_against_us") or feat.get("sack_taken"):
        score += 1
        reasons.append("Negative outcome")
    # anomalies
    if (feat.get("player_count_p50") and feat["player_count_p50"] < 11) or feat.get("spacing_anomaly"):
        score += 1
        reasons.append("Player count/spacing anomaly")
    return {"score": score, "reasons": reasons}


def rank_all(out_dir: str, pb):
    out = Path(out_dir)
    plays = _load_jsonl(out / "plays.jsonl")
    feats = _load_jsonl(out / "features.jsonl")
    by_id = {(p.get("seg_id") or p.get("segment_id")): p for p in plays}
    rank_rows = []
    for f in feats:
        sid = f.get("seg_id") or f.get("segment_id")
        if not sid:
            continue
        s = _score_clip(by_id.get(sid, {}), f, pb)
        rank_rows.append({
            "seg_id": sid,
            "score": s["score"],
            "reasons": s["reasons"],
            "start_s": by_id.get(sid, {}).get("start_s"),
            "end_s": by_id.get(sid, {}).get("end_s"),
            "clip": f"highlights/{sid}.mp4",
        })
    rank_rows.sort(key=lambda x: (-x["score"], x.get("start_s", 0)))
    (out / "review").mkdir(exist_ok=True)
    with (out / "review" / "review_rankings.jsonl").open("w") as w:
        for r in rank_rows:
            w.write(json.dumps(r) + "\n")
    print(f"[review_ranker] wrote {out/'review'/'review_rankings.jsonl'} with {len(rank_rows)} rows")
