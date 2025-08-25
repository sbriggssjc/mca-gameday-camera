from __future__ import annotations

import hashlib
from typing import Any, Dict, List

MIN_CONF = 0.85            # up from 0.80; require stronger signal
REPEAT_PENALTY = 0.15      # if last N predictions identical, subtract
REPEAT_WINDOW = 3
UNKNOWN_FLOOR = 0.60       # below this, label "Unknown"

_last_preds: List[Dict[str, float]] = []

FORMATION_FAMILY_PRIOR = {
    "Trips Left":  {"F Stick","Jet Sweep","Quick Screen","Flood","F Screen","Flare Boot","8 Option"},
    "Trips Right": {"F Stick","Jet Sweep","Quick Screen","Flood","F Screen","Flare Boot","8 Option"},
    "Reo":         {"F Stick","Flood","Quick Screen","Jet Sweep"},
    "Leo":         {"F Stick","Flood","Quick Screen","Jet Sweep"},
    "Rit":         {"Dive","Sweep","F Counter","Power R","Jet Sweep","8 Option","Flare Boot","F Screen"},
    "Lit":         {"Dive","Sweep","F Counter","Power L","Jet Sweep","8 Option","Flare Boot","F Screen"},
}

def score_play(features: Dict[str, Any], play: Dict[str, Any]) -> float:
    name = play.get("name", "")
    h = int(hashlib.sha1(name.encode()).hexdigest(), 16)
    return (h % 100) / 100.0

def model_predict_candidates(frame_features: Dict[str, Any], playbook_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for play in playbook_cfg.get("plays", []):
        candidates.append({
            "name": play.get("name", ""),
            "family": play.get("family", ""),
            "score": score_play(frame_features, play),
        })
    return candidates

def classify_play(frame_features: Dict[str, Any], formation: str, playbook_cfg: Dict[str, Any]) -> Dict[str, Any]:
    candidates = model_predict_candidates(frame_features, playbook_cfg)

    families_ok = FORMATION_FAMILY_PRIOR.get(formation)
    if families_ok:
        candidates = [c for c in candidates if (c.get("family") in families_ok or c.get("name") in families_ok)]

    total = sum(max(c["score"], 0.0) for c in candidates) or 1.0
    for c in candidates:
        c["conf"] = c["score"] / total

    if _last_preds and len(_last_preds) >= REPEAT_WINDOW:
        all_same = len({p["name"] for p in _last_preds[-REPEAT_WINDOW:]}) == 1
        if all_same:
            top_name = _last_preds[-1]["name"]
            for c in candidates:
                if c["name"] == top_name:
                    c["conf"] = max(0.0, c["conf"] - REPEAT_PENALTY)

    best = max(candidates, key=lambda x: x["conf"]) if candidates else None

    if not best or best["conf"] < MIN_CONF:
        conf = float(best["conf"]) if best else 0.0
        if conf < UNKNOWN_FLOOR:
            conf = 0.0
        out = {"name": "Unknown", "confidence": conf, "candidates": candidates, "family": ""}
    else:
        out = {"name": best["name"], "confidence": best["conf"], "candidates": candidates, "family": best.get("family", "")}

    _last_preds.append({"name": out["name"], "confidence": out["confidence"]})
    if len(_last_preds) > 32:
        _last_preds.pop(0)

    return out

__all__ = ["classify_play"]
