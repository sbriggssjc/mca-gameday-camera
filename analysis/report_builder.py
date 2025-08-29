from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence

from collections import Counter

from reporting.generate_report import (
    build_join,
    summarize,
    timeline_rows,
    _load_jsonl,
)

from .io_utils import ensure_dir, write_json
from .segmentation import Segment
from . import report as _legacy_report


def build(
    out_dir: Path,
    metadata_path: Path,
    segments: Sequence[Segment],
    formations: Sequence[str],
    play_matches: Sequence[Dict[str, object]],
    grades_path: Path,
) -> None:
    """Generate dashboards and summary report files."""

    dashboards = out_dir / "dashboards"
    ensure_dir(dashboards)

    joined = build_join(out_dir)
    preds = _load_jsonl(out_dir / "play_predictions.jsonl")
    unk = [p for p in preds if p.get("predicted_play") == "UNKNOWN"]
    reasons = Counter((p.get("why") or "unknown") for p in unk)
    (
        play_counts,
        avg_grade,
        median_conf,
        unknown_count,
        ungradables,
        total,
    ) = summarize(joined)

    summary = {
        "play_count": len(joined),
        "plays": dict(play_counts),
        "median_confidence": median_conf,
        "unknown_predictions": unknown_count,
    }
    write_json(dashboards / "summary.json", summary)

    # Classifier weakness stats from plays_index.csv
    weak_count = 0
    avg_conf = 0.0
    confs: list[float] = []
    index_csv = out_dir / "plays_index.csv"
    if index_csv.exists():
        with index_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    conf = float(row.get("clf_top1_conf", 0.0))
                    confs.append(conf)
                    if int(row.get("clf_weak_flag", 0)):
                        weak_count += 1
                except ValueError:
                    continue
    if confs:
        avg_conf = sum(confs) / len(confs)

    rows = timeline_rows(joined)
    with (dashboards / "timeline.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "Start", "End", "Duration", "Tag", "Note"])
        for r in rows:
            writer.writerow([r["num"], r["start"], r["end"], r["dur"], r["tag"], r["note"]])

    md_lines = ["# Game Report", "", f"Total plays: {len(joined)}", ""]
    md_lines.append(f"Median confidence: {median_conf:.2f}")
    md_lines.append(f"Unknown predictions: {unknown_count}")
    md_lines.append("")
    md_lines.append("## Classifier Weakness")
    md_lines.append("|Weak segments|Avg confidence|")
    md_lines.append("|---|---|")
    md_lines.append(f"|{weak_count}|{avg_conf:.2f}|")
    md_lines.append("")

    md_lines.append("## Unknown Root Causes")
    if unk:
        for k, v in reasons.items():
            md_lines.append(f"- {k}: {v}")
    else:
        md_lines.append("- none")
    md_lines.append("")

    md_lines.append("## Plays Detected")
    for name, count in play_counts.items():
        md_lines.append(f"- {name}: {count}")
    md_lines.append("")

    md_lines.append("## Defensive Grade")
    if total and ungradables / total > 0.4:
        md_lines.append("Avg defense: N/A (insufficient gradable plays)")
    elif avg_grade is not None:
        md_lines.append(f"Average: {avg_grade:.2f}")
    else:
        md_lines.append("Avg defense: N/A")
    md_lines.append("")

    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf8")
    _legacy_report._write_dummy_pdf("summary", str(out_dir / "report.pdf"))
