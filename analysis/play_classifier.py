from typing import Dict, List, Any, Tuple

UNKNOWN_THRESH = 0.45           # below this => "Unknown"
FORMATION_BOOST = 1.10          # light bump for formation-compatible plays
MAX_CANDIDATES = 5

# Example lookup of which plays are compatible with which formations.
# Expand as needed; unknown formations won't filter.
FORMATION_PLAY_FILTER: Dict[str, List[str]] = {
    "Trips Left":  ["Leo F Stick","Lit Jet Sweep","Lit Flare Boot","Lit F Screen","Lit 8 Option"],
    "Trips Right": ["Rit Jet Sweep","Rit Flare Boot","Rit F Screen","Rit 8 Option","Leo F Stick"],
}


def _model_scores_for_segment(seg: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Return [(play_name, score_0to1), ...] sorted desc. 
    Replace this stub with the real model call as appropriate.
    """
    # If you already have model outputs on the segment, adapt here.
    # But always normalize to 0..1 floats and sort desc.
    raw = seg.get("_model_logits") or []
    pairs = [(name, float(score)) for name, score in raw]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def _rescore_by_formation(cands: List[Tuple[str, float]], formation: str) -> List[Tuple[str, float]]:
    if not cands:
        return cands
    legal = set(FORMATION_PLAY_FILTER.get(formation or "", []))
    rescored: List[Tuple[str, float]] = []
    for name, score in cands:
        bump = FORMATION_BOOST if name in legal else 1.0
        rescored.append((name, min(1.0, score * bump)))
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored


def classify_plays(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each detected segment, return a dict with:
      - 'play_family': str (or 'Unknown')
      - 'playcall_confidence': float in [0,1]
      - 'candidates': list of {'name': str, 'score': float}
    Safe under empty/low-confidence outputs.
    """
    results: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments, start=1):
        formation = seg.get("formation") or ""
        cands = _model_scores_for_segment(seg)
        cands = _rescore_by_formation(cands, formation)

        # truncate + standardize candidate objects
        cand_objs = [{"name": n, "score": float(s)} for n, s in cands[:MAX_CANDIDATES]]

        top_name, top_score = ("Unknown", 0.0)
        if cand_objs:
            top_name, top_score = cand_objs[0]["name"], cand_objs[0]["score"]

        play_family = top_name if top_score >= UNKNOWN_THRESH else "Unknown"

        result: Dict[str, Any] = {
            "play_id": seg.get("play_id") or seg.get("id") or f"PLAY_{i:03d}",
            "t0": seg.get("t0", 0.0),
            "t1": seg.get("t1", 0.0),
            "snap": seg.get("snap"),
            "whistle": seg.get("whistle"),
            "clip_path": seg.get("clip_path", ""),
            "formation": formation or "Unknown",
            "formation_confidence": float(seg.get("formation_confidence", 0.0)),
            "play_family": play_family,
            "playcall_confidence": float(top_score),
            "outcome": seg.get("outcome", ""),
            "clip_duration": float(seg.get("clip_duration", max(0.0, (seg.get("t1", 0.0) or 0.0) - (seg.get("t0", 0.0) or 0.0)))),
            "candidates": cand_objs,
        }
        results.append(result)
    return results
