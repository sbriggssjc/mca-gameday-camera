import csv
import json
from pathlib import Path


def write_play_index(outdir: Path, play_rows: list[dict]):
    with open(outdir / "plays_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(play_rows[0].keys()))
        w.writeheader()
        w.writerows(play_rows)


def write_coach_summary(outdir: Path, game_meta: dict, plays: list[dict], players: dict):
    # Minimal HTML report + CSVs; keep simple and readable
    (outdir / "summaries").mkdir(parents=True, exist_ok=True)
    (outdir / "summaries" / "readme.txt").write_text(
        "This folder contains coach-ready summaries: plays_index.csv, player_grades.csv, and per-play JSON."
    )
