import json
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def _score_clip(play: Dict[str, Any], feat: Dict[str, Any], pb) -> Dict[str, Any]:
    score, reasons = 0.0, []

    called = (play.get("called_play") or play.get("play_name") or "").lower()
    off = pb.get_offense_play_by_name(called) if (pb and called) else None

    detected_family = (feat.get("offense_family") or "").lower()
    if off and detected_family and detected_family != (off.get("family", "").lower()):
        score += 3
        reasons.append(f"Mismatch: expected {off.get('family')} got {detected_family}")

    detected_form = (feat.get("formation") or "").title()
    if off and detected_form and detected_form != off.get("formation"):
        score += 2
        reasons.append(
            f"Formation mismatch: expected {off.get('formation')} got {detected_form}"
        )

    if feat.get("edge_broken_right") or feat.get("edge_broken_left"):
        score += 2
        reasons.append("Edge/contain lost")

    hit_gap = (feat.get("run_hit_gap") or "").upper()
    if hit_gap in {"A", "B", "C", "D"} and feat.get("gap_unfilled", False):
        score += 2
        reasons.append(f"Unfilled gap: {hit_gap}")

    if feat.get("explosive_gain_allowed") or feat.get("explosive_gain_for"):
        score += 2
        reasons.append("Explosive play")
    if feat.get("tfl_against_us") or feat.get("sack_taken"):
        score += 1
        reasons.append("Negative outcome")

    if (feat.get("player_count_p50") and feat["player_count_p50"] < 11) or feat.get(
        "spacing_anomaly"
    ):
        score += 1
        reasons.append("Player count/spacing anomaly")

    return {"score": score, "reasons": reasons}


def rank_all(out_dir: str, pb) -> None:
    out = Path(out_dir)
    plays = _load_jsonl(out / "plays.jsonl")
    feats = _load_jsonl(out / "features.jsonl")
    by_id = {(p.get("seg_id") or p.get("segment_id")): p for p in plays}
    rows = []
    for f in feats:
        sid = f.get("seg_id") or f.get("segment_id")
        if not sid:
            continue
        s = _score_clip(by_id.get(sid, {}), f, pb)
        rows.append(
            {
                "seg_id": sid,
                "score": s["score"],
                "reasons": s["reasons"],
                "start_s": by_id.get(sid, {}).get("start_s"),
                "end_s": by_id.get(sid, {}).get("end_s"),
                "clip": f"highlights/{sid}.mp4",
            }
        )
    rows.sort(key=lambda x: (-x["score"], x.get("start_s") or 0))
    (out / "review").mkdir(exist_ok=True)
    with (out / "review" / "review_rankings.jsonl").open("w") as w:
        for r in rows:
            w.write(json.dumps(r) + "\n")
    print(f"[review_ranker] wrote {out/'review'/'review_rankings.jsonl'} ({len(rows)} rows)")


__all__ = ["rank_all"]

