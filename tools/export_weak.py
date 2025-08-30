#!/usr/bin/env python3
"""Export weakly classified clips for labeling/retraining.

Example:
    python3 tools/export_weak.py --runs output/games/*__latest --out dataset/weak_samples
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
from pathlib import Path


def export_weak(runs: list[str], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    for pattern in runs:
        for run in glob.glob(pattern):
            run_path = Path(run)
            csv_path = run_path / "plays_index.csv"
            if not csv_path.exists():
                continue
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    weak = False
                    for key in ("clf_weak_flag", "formation_weak"):
                        val = row.get(key)
                        if val and val not in ("0", "", "0.0", 0, 0.0):
                            weak = True
                            break
                    if not weak:
                        continue
                    clip_rel = row.get("clip_path")
                    if not clip_rel:
                        continue
                    clip_src = Path(clip_rel)
                    if not clip_src.is_absolute():
                        clip_src = run_path / clip_src
                    if not clip_src.exists():
                        continue
                    dest_clip = out_dir / f"{run_path.name}_{clip_src.name}"
                    shutil.copy2(clip_src, dest_clip)
                    meta = {
                        "run": run_path.name,
                        "source": str(clip_src),
                        "t0": row.get("t0"),
                        "t1": row.get("t1"),
                        "snap": row.get("snap"),
                        "whistle": row.get("whistle"),
                        "clf_top1": row.get("clf_top1"),
                        "clf_top1_conf": row.get("clf_top1_conf"),
                        "clf_top3": row.get("clf_top3"),
                        "candidates": row.get("candidates"),
                    }
                    with dest_clip.with_suffix(".json").open("w") as sf:
                        json.dump(meta, sf, indent=2)
                    manifest_rows.append(
                        {
                            "run": run_path.name,
                            "clip": dest_clip.name,
                            "t0": row.get("t0"),
                            "t1": row.get("t1"),
                            "clf_top1": row.get("clf_top1"),
                            "clf_top1_conf": row.get("clf_top1_conf"),
                            "clf_top3": row.get("clf_top3"),
                            "candidates": row.get("candidates"),
                        }
                    )

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as mf:
        fieldnames = [
            "run",
            "clip",
            "t0",
            "t1",
            "clf_top1",
            "clf_top1_conf",
            "clf_top3",
            "candidates",
        ]
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"exported {len(manifest_rows)} weak clips to {out_dir}")
    return len(manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export weak clips for labeling")
    parser.add_argument("--runs", nargs="+", required=True, help="Glob for run directories")
    parser.add_argument("--out", required=True, type=Path, help="Destination directory")
    args = parser.parse_args()
    export_weak(args.runs, args.out)


if __name__ == "__main__":
    main()
