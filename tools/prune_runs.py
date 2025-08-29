#!/usr/bin/env python3
"""Prune old game runs, keeping only the latest N per game.

Example:
    python3 tools/prune_runs.py --games-dir output/games --keep 3
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


def prune_runs(games_dir: Path, keep: int) -> None:
    groups: dict[str, list[Path]] = defaultdict(list)

    if not games_dir.exists():
        return

    for entry in games_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if "__" not in name:
            continue
        base, _, suffix = name.rpartition("__")
        if suffix == "latest":
            continue
        groups[base].append(entry)

    for base, runs in groups.items():
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for old in runs[keep:]:
            print(f"prune {old}")
            shutil.rmtree(old, ignore_errors=True)
        subprocess.run(["scripts/update_latest_symlinks.sh", base], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune old game runs")
    parser.add_argument("--games-dir", default="output/games", type=Path)
    parser.add_argument("--keep", default=3, type=int)
    args = parser.parse_args()

    prune_runs(args.games_dir, args.keep)


if __name__ == "__main__":
    main()
