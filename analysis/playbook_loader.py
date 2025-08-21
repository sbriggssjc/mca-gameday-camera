"""Robust playbook loader accepting multiple schemas and paths."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
from tools.json_io import load_json_safe


def _try_paths(p: str) -> List[Path]:
    cand = [Path(p)]
    # common fallbacks
    cand.append(Path("playbooks") / p)
    cand.append(Path("playbooks") / Path(p).name)
    return [c for c in cand if c.exists()]


def load_playbook(playbook_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load a playbook from disk.

    Returns a tuple of ``(raw_playbook, plays_list)`` and prints a helpful log
    message describing what was loaded.  Supports a few common JSON schemas:

    ``{"plays": [...]}``
        Flat list at top-level.
    ``{"offense": {"plays": [...]}, "defense": {...}}``
        Split offense/defense sections.
    ``{"sections": {"offense": {"plays": [...]}}}``
        Nested sections variant.
    """

    candidates = _try_paths(playbook_path)
    if not candidates:
        print(f"[playbook] ERROR: not found: {playbook_path} (also tried playbooks/ variants)")
        return {}, []
    pb_path = candidates[0]
    raw = load_json_safe(pb_path)
    if raw is None:
        print(f"[playbook] ERROR: failed to parse JSON at {pb_path}")
        return {}, []

    plays: List[Dict[str, Any]] = []
    if isinstance(raw, dict) and isinstance(raw.get("plays"), list):
        # schema 1
        plays = raw["plays"]
    elif isinstance(raw.get("offense"), dict) and isinstance(raw["offense"].get("plays"), list):
        # schema 2
        plays = raw["offense"]["plays"]
    elif isinstance(raw.get("sections"), dict):
        # schema 3 (split sections)
        off = raw["sections"].get("offense") or raw["sections"].get("Offense")
        if isinstance(off, dict) and isinstance(off.get("plays"), list):
            plays = off["plays"]

    if not plays:
        keys = list(raw.keys()) if isinstance(raw, dict) else type(raw)
        print(f"[playbook] WARNING: 0 plays parsed from {pb_path}. Top-level keys: {keys}")
    else:
        print(f"[playbook] OK: loaded {len(plays)} plays from {pb_path.name}")
    return raw, plays


__all__ = ["load_playbook"]

