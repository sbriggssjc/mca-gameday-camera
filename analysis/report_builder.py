from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence

from reporting.generate_report import (
    build_joined_rows,
    summarize,
    timeline_rows,
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

    joined = build_joined_rows(out_dir)
    formation_counts, play_counts, _, _ = summarize(joined)

    summary = {
        "play_count": len(joined),
        "formations": dict(formation_counts),
        "plays": dict(play_counts),
    }
    write_json(dashboards / "summary.json", summary)

    rows = timeline_rows(joined)
    with (dashboards / "timeline.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "Start", "End", "Duration", "Tag", "Note"])
        for r in rows:
            writer.writerow([r["num"], r["start"], r["end"], r["dur"], r["tag"], r["note"]])

    md_lines = ["# Game Report", "", f"Total plays: {len(joined)}", ""]

    md_lines.append("## Formations Used")
    for name, count in formation_counts.items():
        md_lines.append(f"- {name}: {count}")
    md_lines.append("")

    md_lines.append("## Plays Detected")
    for name, count in play_counts.items():
        md_lines.append(f"- {name}: {count}")
    md_lines.append("")

    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf8")
    _legacy_report._write_dummy_pdf("summary", str(out_dir / "report.pdf"))
