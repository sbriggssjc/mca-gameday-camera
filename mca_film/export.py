"""Export helpers for analysis deliverables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .models import CoachSummary, PlayAnalysis


def export_coach_summary(analyses: Iterable[PlayAnalysis]) -> Path:
    """Export a very small CSV summarising average grades per player."""

    player_totals: Dict[str, List[float]] = {}
    for play in analyses:
        for pid, grade in play.assignments.items():
            player_totals.setdefault(pid, []).append(grade.grade)
    averages = {pid: sum(g) / len(g) for pid, g in player_totals.items()}
    out_path = Path("out") / "reports" / "coaches_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["player_id", "avg_grade"])
        for pid, avg in sorted(averages.items()):
            writer.writerow([pid, f"{avg:.2f}"])
    return out_path


def export_player_clips(analyses: Iterable[PlayAnalysis], player_id: str) -> Dict[str, Path]:
    """Create placeholder positive/correction files for a player."""

    base = Path("out") / "players" / player_id
    base.mkdir(parents=True, exist_ok=True)
    pos = base / "positives.mp4"
    corr = base / "corrections.mp4"
    # We do not actually render video; just touch the files.
    pos.touch()
    corr.touch()
    return {"positives": pos, "corrections": corr}


def export_highlights(analyses: Iterable[PlayAnalysis]) -> Path:
    """Create a placeholder highlights reel file."""

    out_path = Path("out") / "highlights" / "mca_highlights.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch()
    return out_path
