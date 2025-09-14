"""Very small heuristic role labelling helpers.

This is a drastically simplified placeholder that assigns fixed role labels
based on horizontal ordering of players at the start of the play.  The intent is
simply to provide deterministic labels for the aerial renderer when jersey
numbers are unknown.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def guess_roles(players: List[Tuple[str, float, float]]) -> Dict[str, str]:
    """Return mapping ``track_id -> role``.

    ``players`` is a list of ``(track_id, x, y)`` tuples using field coordinates
    in yards.  The player with the smallest ``x`` (left-most) is labelled ``X``,
    the largest ``x`` becomes ``Z`` and the remaining players are assigned
    ``A``, ``B``, ... alphabetically.  This naive approach is sufficient for unit
    tests and can be replaced with a more sophisticated formation-based method
    later on.
    """

    if not players:
        return {}
    players_sorted = sorted(players, key=lambda p: p[1])
    roles: Dict[str, str] = {}
    if players_sorted:
        roles[players_sorted[0][0]] = "X"
    if len(players_sorted) > 1:
        roles[players_sorted[-1][0]] = "Z"
    middle = players_sorted[1:-1]
    role_letters = ["A", "B", "C", "D", "E", "F", "G"]
    for (tid, _x, _y), role in zip(middle, role_letters):
        roles[tid] = role
    return roles
