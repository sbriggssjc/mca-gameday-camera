"""Placeholder video clipping helpers."""
from __future__ import annotations

import os
from typing import Dict, Any, Iterable


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"")


def export_clips(
    grades: Iterable[Dict[str, Any]],
    out_dir: str,
    corrections: bool = False,
    wins: bool = False,
    highlights: bool = False,
) -> None:
    """Generate empty video files representing various clip categories."""

    players = set()
    for play in grades:
        players.update(play["players"].keys())

    if corrections:
        if not players:
            players = {1}
        for p in players:
            _touch(os.path.join(out_dir, "clips", "corrections", f"{p}_corrections.mp4"))
    if wins:
        if not players:
            players = {1}
        for p in players:
            _touch(os.path.join(out_dir, "clips", "wins", f"{p}_wins.mp4"))
    if highlights:
        _touch(os.path.join(out_dir, "clips", "highlights", "team_highlights.mp4"))
