"""Playbook assignment utilities."""

from __future__ import annotations

import json
from typing import Dict, List


def load_playbook(path: str | None) -> List[Dict[str, str]]:
    """Load a simplified playbook file.

    The expected format is a JSON object with a top level key ``plays``
    containing a list of play definitions.  Each play definition should at
    least provide a ``name`` and ``formation`` field.  The production
    repository contains a much richer structure but that is unnecessary for
    the unit tests in this kata.
    """

    if not path:
        return []
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "plays" in data:
        return data["plays"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognised playbook format")


def assignments_for_play(play_name: str, playbook: List[Dict[str, str]]) -> Dict[str, str]:
    """Return assignment mapping for ``play_name`` if present."""

    for play in playbook:
        if play.get("name") == play_name:
            return play.get("assignments", {})
    return {}
