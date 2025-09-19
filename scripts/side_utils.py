"""Shared helpers for determining which side of the ball a team is on."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _normalize(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def side_for(team_name: str, play: Dict[str, Any]) -> Optional[str]:
    """Return the side ("offense"/"defense") for ``team_name`` in ``play``.

    The legacy ``play["side"]`` field represents the opponent (Jenks) side for
    most historical data sets.  When explicit ``jenks_side``/``metro_side`` keys
    are available we prefer them, falling back to the legacy field so that older
    exports keep working.
    """

    team = (_normalize(team_name) or "")
    js = _normalize(play.get("jenks_side")) if isinstance(play, dict) else None
    ms = _normalize(play.get("metro_side")) if isinstance(play, dict) else None
    legacy = _normalize(play.get("side")) if isinstance(play, dict) else None

    if "jenks" in team:
        return js or legacy

    if any(alias in team for alias in ("metro", "mca", "eagles")):
        if ms:
            return ms
        if js == "offense":
            return "defense"
        if js == "defense":
            return "offense"
        return legacy

    return legacy
