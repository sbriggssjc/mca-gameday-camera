"""Minimal play classifier (heuristic) — replaces Unknown with a sane guess."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, Tuple

# Cache playbook-derived buckets
_PLAY_FAMILIES: Dict[str, set[str]] | None = None


def _load_play_buckets(playbook_path: str | None) -> Dict[str, set[str]]:
    global _PLAY_FAMILIES
    if _PLAY_FAMILIES is not None:
        return _PLAY_FAMILIES
    # Default candidates (keep in sync with playbooks/__init__.py)
    candidates = [
        "playbooks/mca_5th_playbook.json",
        "mca_5th_playbook.json",
        "playbooks/mca_5th_v2.json",
        "playbooks/mca_full_playbook_final.json",
        "mca_5th_v2.json",
        "mca_full_playbook_final.json",
    ]
    paths = [Path(playbook_path)] if playbook_path else [Path(p) for p in candidates]
    data = None
    for p in paths:
        if p and p.exists():
            try:
                data = json.loads(p.read_text())
                break
            except Exception:
                pass
    buckets: Dict[str, set[str]] = {
        "reo_leo": set(),     # pass‑leaning
        "rit_lit": set(),     # run‑leaning
        "rend_lend": set(),   # run‑leaning
        "other": set(),
    }
    if isinstance(data, dict) and "plays" in data:
        for pl in data["plays"]:
            name = (pl.get("name") or "").strip()
            if not name:
                continue
            lname = name.lower()
            if "reo" in lname or "leo" in lname:
                buckets["reo_leo"].add(name)
            elif "rit" in lname or "lit" in lname:
                buckets["rit_lit"].add(name)
            elif "rend" in lname or "lend" in lname:
                buckets["rend_lend"].add(name)
            else:
                buckets["other"].add(name)
    _PLAY_FAMILIES = buckets
    return buckets


def classify_play(
    segment: Any,
    *,
    detected_formation: str | None,
    formation_confidence: float | None,
    playbook_path: str | None = None,
) -> Tuple[str | None, float]:
    """
    Returns (play_name, confidence). Heuristic:
      - map formation text to family bucket, sample a canonical play name
      - confidence = 0.6 base, +0.2 if formation_conf >= 0.75
    """
    form = (detected_formation or "").lower()
    conf = float(formation_confidence or 0.0)
    buckets = _load_play_buckets(playbook_path)

    # Choose bucket by formation keywords
    if any(k in form for k in ("reo", "leo", "trips", "spread")):
        choices = buckets["reo_leo"] or buckets["other"]
    elif any(k in form for k in ("rit", "lit", "i‑", "under", "tight")):
        choices = buckets["rit_lit"] or buckets["other"]
    elif any(k in form for k in ("rend", "lend")):
        choices = buckets["rend_lend"] or buckets["other"]
    else:
        choices = buckets["other"]

    if not choices:
        return None, 0.0

    # Deterministic pick: smallest lexicographic (stable output)
    guess = sorted(choices)[0]
    base = 0.60
    if conf >= 0.75:
        base = 0.80
    return guess, base


__all__ = ["classify_play"]

