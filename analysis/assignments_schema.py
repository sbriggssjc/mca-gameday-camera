"""Utilities for handling multiple playbook schemas.

This module provides schema detection and normalisation helpers so that
playbooks authored in a variety of JSON structures can be ingested and
converted into a canonical in-memory representation used by the
pipeline.  The logic is intentionally lightweight – it only performs the
minimal validation needed for the unit tests and outputs helpful error
messages when required fields are missing.
"""

from __future__ import annotations

from typing import Dict, Any

import logging

logger = logging.getLogger(__name__)


CANONICAL_TEMPLATE: Dict[str, Any] = {
    "offense": {
        "formations": {},
        "plays": [],
    },
    "defense": {
        "base": "",
        "calls": {},
        "positions": {},
        "checks": {},
    },
}


def _normalise_key(key: str) -> str:
    """Return a lower case, whitespace trimmed key."""

    return key.strip().lower()


def detect_schema(playbook: Dict[str, Any]) -> str:
    """Detect the structure of ``playbook``.

    Parameters
    ----------
    playbook:
        Raw playbook dictionary loaded from JSON.

    Returns
    -------
    str
        A simple label describing the schema.
    """

    keys = {_normalise_key(k) for k in playbook.keys()}
    if "offense" in keys or "defense" in keys:
        return "split_sections"
    if "plays" in keys or any(k.endswith("_plays") for k in keys):
        return "flat_lists"
    return "minimal"


def _warn_unknown(path: str, data: Dict[str, Any], allowed: set[str]) -> None:
    for key in data.keys():
        if _normalise_key(key) not in allowed:
            logger.warning("Unknown key '%s' at %s", key, path)


def _normalise_offense_from_play_list(plays: Any) -> list[Dict[str, Any]]:
    normalised = []
    if not isinstance(plays, list):
        raise ValueError("offense.plays should be a list of plays")
    for idx, play in enumerate(plays):
        if not isinstance(play, dict):
            logger.warning("Skipping play at index %s: not a dict", idx)
            continue
        name = play.get("name")
        formation = play.get("formation")
        if not name or not formation:
            logger.warning(
                "Skipping play at index %s due to missing required fields", idx
            )
            continue
        normalised.append(play)
    return normalised


def normalise(playbook: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise ``playbook`` into the canonical representation."""

    schema = detect_schema(playbook)
    logger.info("Detected playbook schema: %s", schema)

    canonical = {
        "offense": {"formations": {}, "plays": []},
        "defense": {"base": "", "calls": {}, "positions": {}, "checks": {}},
    }

    if schema == "split_sections":
        offense = playbook.get("offense", playbook.get("Offense", {}))
        defense = playbook.get("defense", playbook.get("Defense", {}))

        canonical["offense"]["formations"] = offense.get("formations", {})
        plays = offense.get("plays", offense.get("Plays", []))
        canonical["offense"]["plays"] = _normalise_offense_from_play_list(plays)

        canonical["defense"]["base"] = defense.get("base", "")
        canonical["defense"]["calls"] = defense.get("calls", {})
        canonical["defense"]["positions"] = defense.get("positions", {})
        canonical["defense"]["checks"] = defense.get("checks", {})
        _warn_unknown("offense", offense, {"formations", "plays", "positions"})
        _warn_unknown(
            "defense", defense, {"base", "calls", "positions", "checks"}
        )
    elif schema == "flat_lists":
        plays = playbook.get("plays") or playbook.get("offense_plays", [])
        formations = playbook.get("formations", {})
        canonical["offense"]["formations"] = formations
        canonical["offense"]["plays"] = _normalise_offense_from_play_list(plays)
        # Defense information is optional in this schema; we simply take known keys
        canonical["defense"]["positions"] = playbook.get(
            "defense_positions", {}
        )
    else:  # minimal / name map
        plays = []
        for name, details in playbook.items():
            if not isinstance(details, dict):
                logger.warning("Skipping play '%s': not a dict", name)
                continue
            details = {k: v for k, v in details.items()}
            details.setdefault("name", name)
            if "formation" not in details:
                logger.warning("Skipping play '%s': missing formation", name)
                continue
            plays.append(details)
        canonical["offense"]["plays"] = plays

    return canonical
