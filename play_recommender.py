"""Play recommendation engine based on recent history and defense looks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PLAYBOOK_PATH = Path("plays/playbook.json")


def _load_playbook() -> List[Dict[str, Any]]:
    """Return playbook entries from :data:`PLAYBOOK_PATH`."""
    if not PLAYBOOK_PATH.exists():
        return []
    try:
        with PLAYBOOK_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            return [dict(name=k, **v) for k, v in data.items() if isinstance(v, dict)]
    except Exception:
        pass
    return []


def recommend_play(recent_plays: List[Dict[str, Any]], defense_look: str | None) -> List[Dict[str, str]]:
    """Return top play recommendations.

    Parameters
    ----------
    recent_plays:
        List of recent play dictionaries. Each should include ``play_name`` and
        ``success`` keys.
    defense_look:
        Detected defensive front, e.g. ``"4-4"`` or ``"5-2"``.

    Returns
    -------
    List[Dict[str, str]]
        Up to three play suggestions sorted by highest score. Each dictionary
        contains ``play_name``, ``wristband_code`` and a short ``reason``.
    """

    playbook = _load_playbook()
    if not playbook:
        return []

    # gather success rates from recent history
    stats: Dict[str, Dict[str, int]] = {}
    for p in recent_plays:
        name = str(p.get("play_name") or p.get("play_type") or "").strip()
        if not name:
            continue
        stat = stats.setdefault(name, {"cnt": 0, "success": 0})
        stat["cnt"] += 1
        if p.get("success"):
            stat["success"] += 1

    last_called = [str(p.get("play_name") or p.get("play_type") or "").strip() for p in recent_plays[-3:]]

    scored: List[tuple[float, Dict[str, Any]]] = []
    for entry in playbook:
        name = str(entry.get("name"))
        info = stats.get(name, {"cnt": 0, "success": 0})
        rate = info["success"] / info["cnt"] if info["cnt"] else 0.0
        score = rate
        if defense_look and defense_look in entry.get("success_against", []):
            score += 0.3
        if name in last_called:
            score -= 0.2
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    recommendations: List[Dict[str, str]] = []
    for score, entry in scored[:3]:
        recommendations.append(
            {
                "play_name": entry.get("name", ""),
                "wristband_code": str(entry.get("wristband_code", "")),
                "reason": f"score {score:.2f}",
            }
        )
    return recommendations


__all__ = ["recommend_play"]
