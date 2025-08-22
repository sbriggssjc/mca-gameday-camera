#!/usr/bin/env python3
"""Update per-player season statistics from highlight clips."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

try:
    from fpdf import FPDF  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FPDF = None  # type: ignore

STAT_FIELDS = ["tds", "blocks", "tackles", "ints", "hustle_tags"]


def parse_clip_name(filename: str) -> tuple[str, str]:
    """Return (label, player) parsed from ``filename``."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return "", ""
    label = parts[0]
    player = parts[1]
    return label, player


def tally_clips(directory: str) -> Dict[str, Dict[str, int]]:
    """Return stats keyed by player for all clips in ``directory``."""
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {f: 0 for f in STAT_FIELDS})
    dir_path = Path(directory)
    if not dir_path.exists():
        raise SystemExit(f"Clips directory not found: {directory}")
    for clip in dir_path.glob("*.mp4"):
        label, player = parse_clip_name(clip.name)
        if not player:
            continue
        key = player
        l = label.lower()
        if "td" in l:
            stats[key]["tds"] += 1
        if "block" in l:
            stats[key]["blocks"] += 1
        if "tackle" in l:
            stats[key]["tackles"] += 1
        if "int" in l:
            stats[key]["ints"] += 1
        if "hustle" in l:
            stats[key]["hustle_tags"] += 1
    return stats


def read_grades(players: Iterable[str], csv_path: str | None) -> Dict[str, float]:
    """Return mapping of player->coach grade."""
    grades: Dict[str, float] = {}
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0].lower() in {"player", "grade"}:
                    continue
                player = row[0].strip()
                try:
                    grade = float(row[1])
                except Exception:
                    grade = 0.0
                grades[player] = grade
    for p in players:
        if p not in grades:
            try:
                val = input(f"Coach grade for {p} (1-10): ")
                grades[p] = float(val)
            except Exception:
                grades[p] = 0.0
    return grades


def append_stats(
    csv_path: Path,
    date: str,
    opponent: str,
    stats: Dict[str, Dict[str, int]],
    grades: Dict[str, float],
) -> None:
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["game_date", "opponent", "player", *STAT_FIELDS, "coach_grade"],
        )
        if new_file:
            writer.writeheader()
        for player, vals in stats.items():
            row = {
                "game_date": date,
                "opponent": opponent,
                "player": player,
                "coach_grade": grades.get(player, 0.0),
            }
            row.update(vals)
            writer.writerow(row)


def season_averages(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Compute average stats per player across all games."""
    totals: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {f: 0.0 for f in STAT_FIELDS + ["coach_grade", "games"]}
    )
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row["player"]
            totals[p]["games"] += 1
            for field in STAT_FIELDS:
                try:
                    totals[p][field] += float(row.get(field, 0))
                except Exception:
                    pass
            try:
                totals[p]["coach_grade"] += float(row.get("coach_grade", 0))
            except Exception:
                pass
    averages: Dict[str, Dict[str, float]] = {}
    for p, vals in totals.items():
        games = vals.pop("games", 1.0) or 1.0
        averages[p] = {k: vals[k] / games for k in vals}
    return averages


def generate_pdf(averages: Dict[str, Dict[str, float]], path: Path) -> None:
    if FPDF is None:
        return
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Season Player Averages", ln=1)
    pdf.set_font("Helvetica", size=12)
    for player, vals in sorted(averages.items()):
        line = (
            f"{player}: TDs {vals['tds']:.1f}, Blocks {vals['blocks']:.1f}, "
            f"Tackles {vals['tackles']:.1f}, INTs {vals['ints']:.1f}, "
            f"Hustle {vals['hustle_tags']:.1f}, Grade {vals['coach_grade']:.2f}"
        )
        pdf.cell(0, 8, line, ln=1)
    pdf.output(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update player stats for a game")
    parser.add_argument("clips", help="Directory of labeled highlight clips")
    parser.add_argument("date", help="Game date YYYY-MM-DD")
    parser.add_argument("opponent", help="Opponent name")
    parser.add_argument(
        "--grades",
        help="CSV file with player,grade columns. Missing players will be prompted",
    )
    parser.add_argument(
        "--stats", default="player_stats.csv", help="Season stats CSV output",
    )
    parser.add_argument(
        "--summary", default="weekly_summary.pdf", help="PDF summary output",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = tally_clips(args.clips)
    if not stats:
        raise SystemExit("No clips found or unable to parse filenames")
    grades = read_grades(stats.keys(), args.grades)
    append_stats(Path(args.stats), args.date, args.opponent, stats, grades)
    if not args.no_pdf:
        avgs = season_averages(Path(args.stats))
        if avgs:
            generate_pdf(avgs, Path(args.summary))


if __name__ == "__main__":
    main()
