from __future__ import annotations

import hashlib
from typing import Any, Dict, List


def score_play(features: Dict[str, Any], play: Dict[str, Any]) -> float:
    """Very small deterministic scoring placeholder."""
    name = play.get("name", "")
    h = int(hashlib.sha1(name.encode()).hexdigest(), 16)
    return (h % 100) / 100.0


def is_compatible(formation_hint: str | None, play: Dict[str, Any]) -> bool:
    name = play.get("name", "").lower()
    hint = (formation_hint or "").lower()
    if "trips" in hint:
        return ("leo" in name) or ("reo" in name) or ("trips" in name)
    if "rit" in hint or "lit" in hint:
        return ("rit" in name) or ("lit" in name)
    return True


def classify_play(features: Dict[str, Any], playbook: Dict[str, Any], formation_hint: str | None = None) -> Dict[str, Any]:
    scored: List[tuple[float, Dict[str, Any]]] = []
    for play in playbook.get("plays", []):
        score = score_play(features, play)
        if formation_hint and not is_compatible(formation_hint, play):
            score *= 0.3
        scored.append((score, play))

    if not scored:
        return {"name": "Unknown", "confidence": 0.0, "candidates": [], "family": ""}

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:3]
    best_score, best_play = top[0]

    THRESH = 0.55
    if best_score < THRESH:
        return {
            "name": "Unknown",
            "confidence": round(float(best_score), 2),
            "candidates": [
                {"name": p["name"], "confidence": round(float(s), 2)} for s, p in top
            ],
            "family": "",
        }

    return {
        "name": best_play.get("name", "Unknown"),
        "confidence": round(float(best_score), 2),
        "candidates": [
            {"name": p["name"], "confidence": round(float(s), 2)} for s, p in top
        ],
        "family": best_play.get("family", ""),
    }


__all__ = ["classify_play"]

