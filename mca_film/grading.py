"""Simplified grading logic."""

from __future__ import annotations

from typing import Dict


DEFAULT_SCALE = [0, 1, 2, 3, 4]


def grade_play(play_tracks: Dict, call_context: Dict, settings: Dict, player_id: str) -> Dict:
    """Return a naive grade for a player.

    The current implementation simply returns a neutral grade of ``2.0`` and
    records no notes.  The function signature mirrors the spec so that future
    development can expand the logic without changing callers.
    """

    scale = settings.get("grading", {}).get("base_scale", DEFAULT_SCALE)
    mid = scale[len(scale) // 2]
    return {"player_id": player_id, "grade": float(mid), "notes": "placeholder"}
