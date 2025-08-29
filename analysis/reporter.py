import csv
import json
from pathlib import Path


# Unified plays_index.csv header
PLAY_INDEX_FIELDS = [
    "seg_id",
    "start_s",
    "end_s",
    "clf_top1",
    "clf_top1_conf",
    "clf_top3",
    "clf_weak_flag",
    "low_activity",
]


def write_play_index(outdir: Path, play_rows: list[dict]):
    """Write plays_index.csv with a consistent header.

    The file is emitted even when ``play_rows`` is empty so downstream tools do
    not fail with a missing CSV.  Any keys not present in ``play_rows`` are
    filled with empty strings.
    """

    path = outdir / "plays_index.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PLAY_INDEX_FIELDS)
        writer.writeheader()
        for row in play_rows:
            writer.writerow({k: row.get(k, "") for k in PLAY_INDEX_FIELDS})


def write_coach_summary(outdir: Path, game_meta: dict, plays: list[dict], players: dict):
    # Minimal HTML report + CSVs; keep simple and readable
    (outdir / "summaries").mkdir(parents=True, exist_ok=True)
    (outdir / "summaries" / "readme.txt").write_text(
        "This folder contains coach-ready summaries: plays_index.csv, player_grades.csv, and per-play JSON."
    )
