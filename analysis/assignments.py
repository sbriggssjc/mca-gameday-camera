"""Playbook assignment utilities."""

from __future__ import annotations

import json
from typing import Dict, List, Any

from . import assignments_schema


def load_playbook(path: str | None) -> Dict[str, Any]:
    """Load and normalise a playbook file.

    The loader accepts a variety of JSON layouts and converts them into a
    canonical in-memory representation.  Unknown keys are ignored with a
    warning while missing required fields trigger :class:`ValueError` with a
    helpful message.
    """

    if not path:
        return assignments_schema.CANONICAL_TEMPLATE.copy()

    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)

    schema = assignments_schema.detect_schema(data)
    playbook = assignments_schema.normalise(data)

    # Emit helpful logging for tests / CLI users
    print(
        f"Detected playbook schema: {schema}\n"
        f"Loaded offense plays: {len(playbook['offense']['plays'])}, "
        f"defense positions: {len(playbook['defense']['positions'])}"
    )
    return playbook


def assignments_for_play(play_name: str, playbook: Dict[str, Any]) -> Dict[str, Any]:
    """Return assignment mapping for ``play_name`` if present."""

    plays = playbook.get("offense", {}).get("plays", [])
    for play in plays:
        if play.get("name") == play_name:
            # Prefer new ``roles`` section but fall back to ``assignments``
            return play.get("roles") or play.get("assignments", {})
    return {}
