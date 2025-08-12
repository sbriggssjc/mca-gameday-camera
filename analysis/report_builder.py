from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from .io_utils import ensure_dir, write_json
from .segmentation import Segment
from . import report as _legacy_report


def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def build(
    out_dir: Path,
    metadata_path: Path,
    formations: Sequence[str],
    segments: Sequence[Segment],
    grades_path: Path,
) -> None:
    """Generate dashboards and summary report files.

    The generated markdown and PDF are intentionally lightweight; they serve
    only to satisfy unit tests while demonstrating integration between the
    various pipeline components.
    """

    dashboards = out_dir / "dashboards"
    ensure_dir(dashboards)

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    formation_counts: dict[str, int] = {}
    for f in formations:
        formation_counts[f] = formation_counts.get(f, 0) + 1

    summary = {
        "play_count": len(segments),
        "formations": formation_counts,
    }
    write_json(dashboards / "summary.json", summary)

    # ------------------------------------------------------------------
    # Timeline CSV
    # ------------------------------------------------------------------
    with (dashboards / "timeline.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "Start", "End", "Duration", "Tag"])
        for idx, seg in enumerate(segments, 1):
            writer.writerow(
                [
                    idx,
                    _fmt_time(seg.start_ts),
                    _fmt_time(seg.end_ts),
                    _fmt_time(seg.duration),
                    formations[idx - 1] if idx - 1 < len(formations) else "Unknown",
                ]
            )

    # ------------------------------------------------------------------
    # Simple markdown / PDF report
    # ------------------------------------------------------------------
    md_lines = ["# Game Report", ""]
    md_lines.append(f"Total plays: {len(segments)}")
    md_lines.append("")

    md_lines.append("## Formations Used")
    for name, count in formation_counts.items():
        md_lines.append(f"- {name}: {count}")
    md_lines.append("")

    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf8")
    _legacy_report._write_dummy_pdf("summary", str(out_dir / "report.pdf"))
