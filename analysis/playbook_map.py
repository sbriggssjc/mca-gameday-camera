"""Playbook indexing utilities."""
from __future__ import annotations

from typing import Dict, List, Tuple


def normalize_key(s: str) -> str:
    return " ".join(s.strip().lower().split())


def build_play_index(playbook: Dict[str, Dict[str, List[Dict[str, str]]]]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
    """
    Returns dict: norm_play_name -> expected_schema
    Also returns formation_index for display, but do not use it to guess plays.
    """
    play_index: Dict[str, Dict[str, str]] = {}
    formation_index: Dict[str, List[str]] = {}

    offense = playbook.get("offense", {})
    plays = offense.get("plays", [])
    for p in plays:
        pname = normalize_key(p.get("name", ""))
        form = normalize_key(p.get("formation", ""))
        play_index[pname] = p
        formation_index.setdefault(form, []).append(pname)
    return play_index, formation_index
