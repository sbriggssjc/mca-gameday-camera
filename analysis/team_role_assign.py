"""Team colour classification and role assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


FORMATION_ROLE_ORDER: Dict[str, List[str]] = {
    "Rit": ["X", "Q", "H"],
    "Lit": ["H", "Q", "X"],
}


def assign_roles(pre_snap: Iterable[Dict[str, float]], formation: str) -> Dict[str, str]:
    """Assign roles based on x position ordering.

    Parameters
    ----------
    pre_snap:
        Iterable of player dictionaries with keys ``player_id`` and ``x``.
    formation:
        Formation key used to look up role ordering.  Only a tiny subset of
        formations are implemented for testing purposes.

    Returns
    -------
    dict
        Mapping of role name to ``player_id``.
    """

    players = sorted(pre_snap, key=lambda p: p["x"])
    roles = FORMATION_ROLE_ORDER.get(formation, [])
    return {role: players[i]["player_id"] for i, role in enumerate(roles) if i < len(players)}
