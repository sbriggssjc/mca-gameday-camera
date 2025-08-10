"""Player grading heuristics."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List


def grade_first_step(expected_angle: float, observed_angle: float, tolerance: float = 30) -> int:
    """Grade the first step direction.

    Returns ``3`` if the absolute difference between the expected and observed
    angles is within ``tolerance`` degrees, otherwise ``0``.  The scoring range
    mirrors the rubric described in the project specification but in a very
    condensed form suitable for unit tests.
    """

    diff = abs((observed_angle - expected_angle + 180) % 360 - 180)
    return 3 if diff <= tolerance else 0


def grade_play(play: Dict[str, str], assignments: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, object]]:
    """Grade each player for a single play.

    Parameters
    ----------
    play:
        Prediction dictionary produced by :mod:`analysis.play_recognizer`.
    assignments:
        Mapping of role to expected metadata.  Only the ``expected_angle`` key
        is honoured in this toy implementation.
    """

    grades: Dict[str, Dict[str, object]] = {}
    for role, info in assignments.items():
        observed = info.get("observed_angle", 0)
        expected = info.get("expected_angle", 0)
        score = grade_first_step(expected, observed)
        grades[role] = {"score": score, "note": "ok" if score == 3 else "miss"}
    return grades
