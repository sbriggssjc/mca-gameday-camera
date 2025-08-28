from __future__ import annotations

"""Simple play classification helpers.

These helpers accept either legacy dict playbooks or ``PlaybookIndex`` style
objects.  The classifier itself is intentionally lightweight; it returns
formation information when present and proposes top-N candidate play names
from the supplied playbook.
"""

from typing import Any, Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Playbook helpers


def _iter_playbook_plays(pb: Any) -> Iterable[Dict[str, Any]]:
    """Yield play dicts from ``pb`` regardless of its structure."""

    if isinstance(pb, dict):
        plays = pb.get("plays", [])
        for p in plays:
            if isinstance(p, dict):
                yield p
        return

    plays = getattr(pb, "plays", None)
    if plays is None and hasattr(pb, "to_dict"):
        plays = pb.to_dict().get("plays", [])
    if plays is None and hasattr(pb, "__dict__"):
        plays = getattr(pb.__dict__, "plays", []) or []

    for p in plays or []:
        if isinstance(p, dict):
            yield p
        else:
            yield {
                "name": getattr(p, "name", ""),
                "formation": getattr(p, "formation", ""),
                "formations": getattr(p, "formations", None),
                "family": getattr(p, "family", ""),
            }


def _tokenize(s: str) -> List[str]:
    return [t for t in s.lower().split() if t]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[lb]


def _name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    tok_a, tok_b = _tokenize(a), _tokenize(b)
    j = _jaccard(tok_a, tok_b)
    lev = _levenshtein(a.lower(), b.lower())
    norm = 1.0 - (lev / max(len(a), len(b), 1))
    return 0.5 * j + 0.5 * max(0.0, norm)


def _best_matches_from_playbook(
    formation: str,
    pb: Any,
    topk: int = 3,
) -> List[Tuple[str, float]]:
    """Return ``topk`` candidate names and scores from ``pb``."""

    cands: List[Tuple[str, float]] = []
    for p in _iter_playbook_plays(pb):
        name = p.get("name") or ""
        if not name:
            continue
        pf = p.get("formation") or ""
        formations = p.get("formations") or []
        if isinstance(formations, str):
            formations = [formations]
        formation_match = False
        if formation and (pf or formations):
            pool = [pf, *formations]
            for f in pool:
                if f and f.lower() in formation.lower() or formation.lower() in f.lower():
                    formation_match = True
                    break
            if not formation_match:
                continue
        sim = _name_similarity(name, name)  # placeholder self-similarity
        score = min(1.0, sim + (0.2 if formation_match else 0.0))
        cands.append((name, score))
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[:topk]


# ---------------------------------------------------------------------------
# Main classifier


def classify_plays(
    segments: List[Dict[str, float]],
    playbook: Any,
    team: str,
) -> List[Dict[str, Any]]:
    """Classify each segment and propose candidate play names."""

    results: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments, 1):
        formation = seg.get("formation", "") or ""
        candidates = _best_matches_from_playbook(formation, playbook, topk=3)
        top_name, top_score = (candidates[0] if candidates else ("", 0.0))
        results.append(
            {
                "play_id": seg.get("id") or seg.get("play_id") or f"PLAY_{i:03d}",
                "formation": formation,
                "formation_confidence": float(seg.get("formation_confidence", 0.0)),
                "play_family": top_name,
                "playcall_confidence": float(top_score),
                "candidates": candidates,
                "outcome": seg.get("outcome", ""),
            }
        )
    return results


__all__ = ["classify_plays"]
