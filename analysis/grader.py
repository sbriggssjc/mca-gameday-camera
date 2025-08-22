"""Player grading heuristics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Any

from .playbook_map import norm


def grade_first_step(expected_angle: float, observed_angle: float, tolerance: float = 30) -> int:
    """Grade the first step direction.

    Returns ``3`` if the absolute difference between the expected and observed
    angles is within ``tolerance`` degrees, otherwise ``0``.  The scoring range
    mirrors the rubric described in the project specification but in a very
    condensed form suitable for unit tests.
    """

    diff = abs((observed_angle - expected_angle + 180) % 360 - 180)
    return 3 if diff <= tolerance else 0


def grade_play_basic(play: Dict[str, str], assignments: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, object]]:
    """Legacy grading of each player for a single play."""

    grades: Dict[str, Dict[str, object]] = {}
    for role, info in assignments.items():
        observed = info.get("observed_angle", 0)
        expected = info.get("expected_angle", 0)
        score = grade_first_step(expected, observed)
        grades[role] = {"score": score, "note": "ok" if score == 3 else "miss"}
    return grades


# ---------------------------------------------------------------------------
# New per-player grading scaffolding
# ---------------------------------------------------------------------------


@dataclass
class PlayerGrade:
    jersey: int
    position: str | None
    assignment: str | None
    effort: float | None
    technique: float | None
    result: float | None
    overall: float | None
    notes: str | None = None


def default_grade_from_tracks(track):
    # Stub: if player flows toward ball and maintains lane -> higher score
    # Use distance-to-ball delta and participation
    return 0.75


def grade_play(
    players_tracks: Dict[int, Any],
    recognized_play: Dict[str, Any] | None,
    assignments_map: Dict[int, Dict[str, Any]],
) -> List[PlayerGrade]:
    grades: List[PlayerGrade] = []
    for pid, tr in players_tracks.items():
        pos = assignments_map.get(pid, {}).get("pos") if assignments_map else None
        g = default_grade_from_tracks(tr)
        grades.append(
            PlayerGrade(
                jersey=pid,
                position=pos,
                assignment=assignments_map.get(pid, {}).get("assignment") if assignments_map else None,
                effort=g,
                technique=g,
                result=g,
                overall=g,
            )
        )
    return grades


def score_edge_contain(players: List[Dict[str, Any]], expected: Dict[str, Any]) -> float:
    if not players:
        return 0.0
    max_x = max(p.get("x", 0.0) for p in players)
    return 1.0 if max_x >= expected.get("edge_target", 0.0) else 0.0


def score_gap_integrity(players: List[Dict[str, Any]], expected: Dict[str, Any]) -> float:
    if not players:
        return 0.0
    return min(1.0, len(players) / float(expected.get("gap_target", 11)))


def score_secondary_depth(players: List[Dict[str, Any]], expected: Dict[str, Any]) -> float:
    if not players:
        return 0.0
    max_y = max(p.get("y", 0.0) for p in players)
    return 1.0 if max_y >= expected.get("depth_target", 0.0) else 0.0


def grade_defense(
    seg: Dict[str, Any],
    pred: Dict[str, Any],
    tracking: Dict[str, Any] | None,
    play_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"overall_defense": None, "subscores": {}, "notes": []}

    pname = norm(pred.get("predicted_play"))
    expected = play_index.get(pname)
    if not expected:
        out["notes"].append("no_expected_schema_for_prediction")
        return out

    players = (tracking or {}).get("players", [])
    if len(players) < 5:
        out["notes"].append("insufficient_tracking")
        return out

    subs = {}
    subs["edge_contain"] = score_edge_contain(players, expected)
    subs["gap_integrity"] = score_gap_integrity(players, expected)
    subs["secondary_depth"] = score_secondary_depth(players, expected)

    weights = {"edge_contain": 0.35, "gap_integrity": 0.45, "secondary_depth": 0.20}
    overall = sum(weights[k] * subs[k] for k in weights if k in subs)
    out["subscores"] = subs
    out["overall_defense"] = round(overall, 2)
    return out
