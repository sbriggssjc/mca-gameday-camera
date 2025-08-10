"""Simplified player grading utilities.

The production system applies a rich set of coaching rubrics to evaluate
player performance.  For unit testing we only need a lightweight stub
that produces deterministic output so downstream reporting and clipping
logic can be exercised.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def grade(
    predictions: Iterable[Dict[str, Any]],
    tracks: Iterable[Any],
    identity_map: Dict[str, str],
    playbook: Dict[str, Any] | None = None,
    weights_path: str | None = None,
) -> List[Dict[str, Any]]:
    """Return placeholder grades for each play.

    Each player starts at 100 points and no mistakes are recorded.  The
    structure mirrors the shape expected by :mod:`analysis.report` and
    :mod:`analysis.clipping`.
    """

    results: List[Dict[str, Any]] = []
    for pred in predictions:
        players = {
            identity_map.get(t.player_id, t.player_id): {
                "grade": 100,
                "notes": [],
                "mistakes": [],
                "positives": [],
            }
            for t in tracks
        }
        results.append(
            {
                "play_id": pred["play_id"],
                "recognized_play": {
                    "name": pred.get("predicted_play"),
                    "confidence": pred.get("confidence", 0.0),
                },
                "players": players,
                "team_highlights": [],
            }
        )
    return results
