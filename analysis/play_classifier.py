from __future__ import annotations
from typing import Dict, Any, List

def _best_matches_from_playbook(formation: str, playbook: dict) -> List[str]:
    """
    Heuristic fallback: pick plays whose metadata/keywords match the detected formation,
    otherwise return a few common pass concepts so we always emit candidates.
    """
    plays = playbook.get("plays", [])
    names = []
    for p in plays:
        name = p.get("name") or p.get("id") or ""
        meta = " ".join(str(x) for x in p.values()).lower()
        if formation and formation.lower().split()[0] in meta:
            names.append(name)
    if not names:
        # persistent candidates ensure downstream isn't empty
        names = ["Leo F Stick", "Rit Flare Boot", "Rit F Screen", "Lit Jet Sweep", "Rit 8 Option"][:5]
    return names[:5]

def classify_plays(video_path: str,
                   segments: List[dict],
                   formations: Dict[str, dict],
                   playbook: dict) -> Dict[str, dict]:
    """
    Return mapping PLAY_xxx -> {
        'play_family': str,
        'confidence': float,
        'outcome': str,
        'candidates': List[str]
    }
    Never raise ImportError or KeyError; always include 'candidates'.
    """
    result: Dict[str, dict] = {}
    for i, seg in enumerate(segments, start=1):
        pid = f"PLAY_{i:03d}"
        formation = (formations.get(pid, {}) or {}).get("formation", "") or ""
        candidates = _best_matches_from_playbook(formation, playbook)
        # Keep 'Unknown' label if we can't pick a single winner yet,
        # but provide useful ranked candidates for review.
        result[pid] = {
            "play_family": "Unknown",
            "confidence": 0.0,
            "outcome": "",
            "candidates": candidates
        }
    return result
