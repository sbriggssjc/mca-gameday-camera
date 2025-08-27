from __future__ import annotations

"""Lightweight play classification utilities.

This module intentionally keeps the implementation simple so that the rest
of the pipeline has a stable API to interact with.  The classifier returns a
list of dictionaries, one per input segment, describing the detected
formation and the most likely play family along with ranked candidates.
"""

from typing import List, Dict, Tuple


def _best_matches_from_playbook(formation: str, playbook: dict) -> List[str]:
    """Very small heuristic to surface plausible plays from the playbook.

    If the playbook contains metadata that mentions the detected formation
    we keep those entries; otherwise we fall back to a handful of common
    concepts so downstream consumers always receive some candidates.
    """

    plays = playbook.get("plays", [])
    names: List[str] = []
    for p in plays:
        name = p.get("name") or p.get("id") or ""
        meta = " ".join(str(v) for v in p.values()).lower()
        if formation and formation.lower().split()[0] in meta:
            names.append(name)
    if not names:
        names = [
            "Leo F Stick",
            "Rit Flare Boot",
            "Rit F Screen",
            "Lit Jet Sweep",
            "Rit 8 Option",
        ]
    return names[:5]


def _log_candidates(play_id: str, cand: List[Tuple[str, float]]) -> None:
    if not cand:
        print(f"[play_classifier] {play_id}: Unknown conf=0.00")
        return
    top = cand[0]
    tops = ", ".join(f"{n}:{s:.2f}" for n, s in cand[:3])
    print(f"[play_classifier] {play_id}: {top[0]} conf={top[1]:.2f}")
    print(f"[play_classifier:candidates] {play_id}: {tops}")


def classify_plays(segments, playbook, team: str) -> list[dict]:
    """
    Returns a list of dicts, one per segment:
      {
        "play_id": "PLAY_001",
        "formation": str,
        "formation_confidence": float,
        "play_family": str,
        "playcall_confidence": float,
        "candidates": List[Tuple[str, float]],
        "outcome": None or str,
      }
    """

    results: List[Dict] = []
    for idx, seg in enumerate(segments, start=1):
        pid = f"PLAY_{idx:03d}"
        formation = seg.get("formation", "Unknown")
        f_conf = float(seg.get("formation_confidence", 0.0))

        names = _best_matches_from_playbook(formation, playbook)
        candidates: List[Tuple[str, float]] = [
            (n, max(0.0, 0.50 - 0.05 * i)) for i, n in enumerate(names)
        ]
        # Ensure sorted descending by score
        candidates.sort(key=lambda x: x[1], reverse=True)

        _log_candidates(pid, candidates)

        play_family = "Unknown"
        play_conf = 0.0
        if candidates and candidates[0][1] >= 0.40:
            play_family = candidates[0][0]
            play_conf = candidates[0][1]

        results.append(
            {
                "play_id": pid,
                "formation": formation or "Unknown",
                "formation_confidence": f_conf,
                "play_family": play_family,
                "playcall_confidence": play_conf,
                "candidates": candidates,
                "outcome": seg.get("outcome"),
            }
        )

    return results


__all__ = ["classify_plays"]

