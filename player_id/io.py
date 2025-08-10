"""Persistence helpers for player profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from schemas import PlayerProfile


def load_roster(path: str = "data/players.json") -> List[PlayerProfile]:
    """Load player profiles from ``path`` if it exists."""

    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return [PlayerProfile(**item) for item in data]


def save_roster(roster: List[PlayerProfile], path: str = "data/players.json") -> None:
    """Persist player profiles to ``path``."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [r.__dict__ for r in roster]
    p.write_text(json.dumps(data, indent=2))
