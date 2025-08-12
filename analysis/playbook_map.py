"""Playbook indexing utilities."""
from __future__ import annotations

from typing import Dict


def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def build_play_index(playbook: Dict) -> Dict[str, Dict]:
    idx: Dict[str, Dict] = {}
    offense = playbook.get("offense", {})
    for p in offense.get("plays", []):
        idx[norm(p.get("name"))] = p
    return idx
