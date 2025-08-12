from __future__ import annotations

from typing import Any, Sequence

from .assignments import Playbook


def classify_formation(playbook: Playbook | None, frames: Sequence[Any], fps: float) -> str:
    """Return a best guess at the offensive formation.

    The heuristic is intentionally trivial: it returns the formation of the
    first offensive play defined in the provided ``playbook``.  If the
    playbook is empty the function falls back to ``"Unknown"``.
    """

    if playbook and playbook.offense_plays:
        return playbook.offense_plays[0].formation
    return "Unknown"
