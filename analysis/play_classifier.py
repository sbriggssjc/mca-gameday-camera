"""Lightweight play classification utilities.

This version is intentionally minimal and focuses on surfacing plausible
play names from a playbook.  The playbook may be a legacy dictionary or a
`PlaybookIndex` style object; the helper functions below iterate through
either form and extract play metadata.  The classifier returns a list of
results with a `playcall` entry that always contains a list of candidate
names.
"""

from __future__ import annotations

from typing import Iterable, Dict, Any, List, Optional


# --- helpers to be robust to dict or PlaybookIndex inputs --------------------
def _iter_play_defs(playbook: Any) -> Iterable[Dict[str, Any]]:
    """Yield play definitions as dictionaries.

    Handles several playbook layouts:

    * Plain dict with key ``"plays"`` (legacy format).
    * Objects exposing ``iter_plays``/``iter``/``__iter__`` returning play
      objects or dictionaries.
    * Objects with ``plays`` or ``by_name`` attributes that are sequences or
      mappings.

    Each yielded dict will contain at least ``name`` and formation-related
    information when available.
    """

    # Case 1: legacy dict format {"plays": [{...}, ...]}
    if isinstance(playbook, dict):
        for p in playbook.get("plays", []):
            if isinstance(p, dict):
                yield p
        return

    # Case 2: object with an iterator method
    for attr in ("iter_plays", "iter", "__iter__"):
        it = getattr(playbook, attr, None)
        if callable(it):
            for p in it():
                if isinstance(p, dict):
                    yield p
                else:
                    yield {
                        "name": getattr(p, "name", None),
                        "formation": getattr(p, "formation", None),
                        "formations": getattr(p, "formations", None),
                        "family": getattr(p, "family", None),
                        "tags": getattr(p, "tags", None),
                    }
            return

    # Case 3: object exposing 'plays' or mapping-like 'by_name'
    for attr in ("plays", "by_name"):
        obj = getattr(playbook, attr, None)
        if obj is None:
            continue
        if isinstance(obj, dict):
            for name, p in obj.items():
                if isinstance(p, dict):
                    if "name" not in p:
                        p = {**p, "name": name}
                    yield p
                else:
                    yield {
                        "name": getattr(p, "name", name),
                        "formation": getattr(p, "formation", None),
                        "formations": getattr(p, "formations", None),
                        "family": getattr(p, "family", None),
                        "tags": getattr(p, "tags", None),
                    }
            return
        elif isinstance(obj, (list, tuple)):
            for p in obj:
                if isinstance(p, dict):
                    yield p
                else:
                    yield {
                        "name": getattr(p, "name", None),
                        "formation": getattr(p, "formation", None),
                        "formations": getattr(p, "formations", None),
                        "family": getattr(p, "family", None),
                        "tags": getattr(p, "tags", None),
                    }
            return

    # Fallback: nothing recognized
    return


def _norm(s: Optional[str]) -> Optional[str]:
    return s.lower().strip() if isinstance(s, str) else None


def _formation_matches(target: Optional[str], cand: Dict[str, Any]) -> float:
    """Soft match score between target formation and candidate entry."""

    if not target:
        return 0.0
    t = _norm(target)
    fs: List[str] = []
    if isinstance(cand.get("formation"), str):
        fs.append(cand["formation"])
    formations = cand.get("formations")
    if formations:
        if isinstance(formations, (list, tuple)):
            fs.extend([f for f in formations if isinstance(f, str)])
        elif isinstance(formations, str):
            fs.append(formations)
    fs_norm = [_norm(f) for f in fs if isinstance(f, str)]
    if t in fs_norm:
        return 1.0
    # Partial credit when both contain 'trips' regardless of side
    if any(("trips" in (f or "") and "trips" in (t or "")) for f in fs_norm):
        return 0.5
    return 0.0


def _best_matches_from_playbook(
    formation: Optional[str], playbook: Any, k: int = 5
) -> List[str]:
    """Return up to ``k`` play names best matching ``formation``."""

    scored: List[tuple[float, str]] = []
    seen: set[str] = set()
    for p in _iter_play_defs(playbook):
        name = p.get("name") if isinstance(p, dict) else None
        if not name or name in seen:
            continue
        seen.add(name)
        score = _formation_matches(formation, p)
        scored.append((score, name))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [n for _, n in scored[:k]]


def classify_plays(segments, playbook, team):
    """Classify plays using simple formation matching.

    Each returned dict contains ``play_id``, ``formation`` and ``playcall`` with
    a confidence score and candidate list.  ``playcall`` is always present to
    simplify downstream logging.
    """

    results: List[Dict[str, Any]] = []
    for seg in segments:
        play_id = seg.get("play_id") or seg.get("id")
        formation = seg.get("formation")
        names = _best_matches_from_playbook(formation, playbook)
        top = names[0] if names else "Unknown"
        conf = 0.0 if top == "Unknown" else (1.0 if formation and names else 0.13)
        results.append(
            {
                "play_id": play_id,
                "playcall": {
                    "name": top,
                    "confidence": conf,
                    "candidates": names[:5],
                },
                "formation": formation,
            }
        )
    return results


__all__ = ["classify_plays"]

