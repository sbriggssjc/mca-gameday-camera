#!/usr/bin/env python3
"""Export labeled play data to a HUDL compatible CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from roster import get_player_name
except Exception:
    def get_player_name(number: int) -> str:  # type: ignore
        return str(number)

Row = Dict[str, str]


HUDL_HEADER = [
    "Start Time",
    "End Time",
    "Off/Def",
    "Down",
    "Distance",
    "Yard Line",
    "Play Type",
    "Formation",
    "Result",
    "Players",
]


def load_rows(csv_path: Path) -> List[Row]:
    """Load all rows from ``csv_path``."""
    rows: List[Row] = []
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")
    return rows


def format_players(value: str | None) -> str:
    """Return formatted player list ``#num Name`` separated by semicolons."""
    if not value:
        return ""
    names: List[str] = []
    for part in value.replace(",", " ").split():
        num = part.lstrip("#")
        if num.isdigit():
            names.append(f"#{num} {get_player_name(int(num))}")
        else:
            names.append(part)
    return ";".join(names)


def hudl_row(row: Row) -> List[str]:
    """Convert input ``row`` to HUDL export fields."""
    start = row.get("start") or row.get("start_time") or row.get("time", "")
    end = row.get("end") or row.get("end_time") or ""
    off_def = row.get("offense") or row.get("off_def") or ""
    down = row.get("down", "")
    distance = row.get("distance") or row.get("yards_to_go") or ""
    yard_line = row.get("yard_line") or row.get("spot") or ""
    play_type = row.get("label") or row.get("play_type") or ""
    formation = row.get("formation") or ""
    result = row.get("result") or (
        f"Gain {row['yards_gained']}" if row.get("yards_gained") else ""
    )
    players = format_players(row.get("players") or row.get("player"))
    return [
        start,
        end,
        off_def,
        down,
        distance,
        yard_line,
        play_type,
        formation,
        result,
        players,
    ]


def write_hudl_csv(rows: Iterable[Row], out_path: Path, player_filter: Optional[str] = None) -> None:
    """Write ``rows`` to ``out_path`` in HUDL format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HUDL_HEADER)
        for row in rows:
            if player_filter and player_filter not in (row.get("player") or row.get("players") or ""):
                continue
            writer.writerow(hudl_row(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HUDL CSV export")
    parser.add_argument(
        "--csv",
        help="Source CSV file",
    )
    parser.add_argument("--week", help="Week number/name")
    parser.add_argument("--opponent", help="Opponent name")
    parser.add_argument("--player", help="Only include rows for a player")
    parser.add_argument("--output-dir", default="hudl_export", help="Export directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = (
        Path(args.csv)
        if args.csv
        else Path("highlight_log.csv") if Path("highlight_log.csv").exists() else Path("scouting_data.csv")
    )

    rows = load_rows(csv_path)

    parts = []
    if args.week:
        parts.append(f"Week_{args.week}")
    if args.opponent:
        parts.append(args.opponent.replace(" ", "_").strip())
    filename = "_".join(parts) or csv_path.stem
    out_path = Path(args.output_dir) / f"{filename}.csv"

    write_hudl_csv(rows, out_path, player_filter=args.player)
    print(f"\u2705 Exported {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
