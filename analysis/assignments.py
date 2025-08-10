"""Playbook assignment utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import assignments_schema


@dataclass
class OffensePlay:
    """Representation of an offensive play."""

    name: str
    formation: str
    motion: str | None = None
    roles: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation for recogniser consumption."""

        data = {"name": self.name, "formation": self.formation}
        if self.motion is not None:
            data["motion"] = self.motion
        if self.roles is not None:
            data["roles"] = self.roles
        return data


@dataclass
class DefensePosition:
    """Representation of a defensive position and responsibilities."""

    name: str
    tech: str | None = None
    gap: str | None = None
    responsibilities: List[str] = field(default_factory=list)


@dataclass
class Playbook:
    """Normalised playbook structure used throughout the pipeline."""

    offense_plays: List[OffensePlay] = field(default_factory=list)
    defense_positions: List[DefensePosition] = field(default_factory=list)
    calls: Dict[str, str] = field(default_factory=dict)
    schema: str = "minimal"


def _load_default() -> Playbook:
    canon = assignments_schema.CANONICAL_TEMPLATE
    return Playbook()


def load_playbook(path: str | None) -> Playbook:
    """Load and normalise a playbook file.

    Supports legacy flat formats and the new ``split_sections`` schema.  The
    returned :class:`Playbook` exposes offensive plays, defensive positions and
    optional defensive calls in a single dataclass.
    """

    if not path:
        return _load_default()

    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)

    schema = assignments_schema.detect_schema(data)

    offense_plays_raw: List[Dict[str, Any]] = []
    defense_positions_raw: Any = []
    calls_raw: Any = {}

    if schema == "split_sections":
        offense = data.get("offense", {})
        defense = data.get("defense", {})
        offense_plays_raw = offense.get("plays", [])
        defense_positions_raw = defense.get("positions", []) or defense.get("base", {}).get("alignment", {})
        calls_raw = defense.get("calls", {})
        if not defense_positions_raw:
            raise ValueError("Playbook missing defense.positions")
    elif schema == "flat_lists":
        offense_plays_raw = data.get("plays") or data.get("offense_plays", [])
        defense_positions_raw = data.get("defense_positions", [])
        calls_raw = data.get("calls", {})
    else:  # minimal
        for name, details in data.items():
            if not isinstance(details, dict):
                continue
            formation = details.get("formation")
            if formation:
                offense_plays_raw.append(
                    {"name": name, "formation": formation, "motion": details.get("motion")}
                )
        defense_positions_raw = data.get("defense_positions", [])
        calls_raw = data.get("calls", {})

    offense_plays = [
        OffensePlay(
            name=p.get("name", ""),
            formation=p.get("formation", ""),
            motion=p.get("motion"),
            roles=p.get("roles") or p.get("assignments"),
        )
        for p in offense_plays_raw
        if p.get("name") and p.get("formation")
    ]

    defense_positions: List[DefensePosition] = []
    if isinstance(defense_positions_raw, dict):
        for name, info in defense_positions_raw.items():
            if not isinstance(info, dict):
                continue
            defense_positions.append(
                DefensePosition(
                    name=name,
                    tech=info.get("tech"),
                    gap=info.get("gap"),
                    responsibilities=info.get("responsibilities", []),
                )
            )
    else:
        for pos in defense_positions_raw:
            if not isinstance(pos, dict):
                continue
            defense_positions.append(
                DefensePosition(
                    name=pos.get("name", ""),
                    tech=pos.get("tech"),
                    gap=pos.get("gap"),
                    responsibilities=pos.get("responsibilities", []),
                )
            )

    if isinstance(calls_raw, list):
        calls = {c.get("cue"): c.get("trigger") for c in calls_raw if c.get("cue")}
    else:
        calls = {str(k): str(v) for k, v in calls_raw.items()}

    playbook = Playbook(
        offense_plays=offense_plays,
        defense_positions=defense_positions,
        calls=calls,
        schema=schema,
    )

    # Emit helpful logging for tests / CLI users
    print(
        f"Detected playbook schema: {schema}\n"
        f"Loaded offense plays: {len(offense_plays)}, defense positions: {len(defense_positions)}"
    )
    return playbook


def assignments_for_play(play_name: str, playbook: Playbook) -> Dict[str, Any]:
    """Return assignment mapping for ``play_name`` if present."""

    for play in playbook.offense_plays:
        if play.name == play_name:
            return play.roles or {}
    return {}
