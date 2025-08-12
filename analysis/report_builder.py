from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Sequence

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
    segments: Sequence[Segment],
    formations: Sequence[str],
    play_matches: Sequence[Dict[str, object]],
    grades_path: Path,
) -> None:
    """Generate dashboards and summary report files."""

    dashboards = out_dir / "dashboards"
    ensure_dir(dashboards)

    # Load grades if available
    grade_map: Dict[int, Dict[str, object]] = {}
    if grades_path.exists():
        with grades_path.open("r", encoding="utf8") as gf:
            for line in gf:
                g = json.loads(line)
                grade_map[int(g.get("play_index", 0))] = g

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    formation_counts: dict[str, int] = {}
    for f in formations:
        formation_counts[f] = formation_counts.get(f, 0) + 1

    play_counts: dict[str, int] = {}
    for m in play_matches:
        name = m.get("name", "Unknown")
        play_counts[name] = play_counts.get(name, 0) + 1

    summary = {
        "play_count": len(segments),
        "formations": formation_counts,
        "plays": play_counts,
    }
    write_json(dashboards / "summary.json", summary)

    # ------------------------------------------------------------------
    # Timeline CSV
    # ------------------------------------------------------------------
    with (dashboards / "timeline.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "Start", "End", "Dur", "Formation", "Play (conf)", "Def Grade"])
        for idx, seg in enumerate(segments, 1):
            formation = formations[idx - 1] if idx - 1 < len(formations) else "Unknown"
            pm = play_matches[idx - 1] if idx - 1 < len(play_matches) else {"name": "Unknown", "confidence": 0.0}
            grade = grade_map.get(idx - 1, {})
            writer.writerow(
                [
                    idx,
                    _fmt_time(seg.start_ts),
                    _fmt_time(seg.end_ts),
                    _fmt_time(seg.duration),
                    formation,
                    f"{pm.get('name')} ({pm.get('confidence', 0.0):.2f})",
                    grade.get("overall_defense", ""),
                ]
            )

    # ------------------------------------------------------------------
    # Simple markdown / PDF report
    # ------------------------------------------------------------------
    md_lines = ["# Game Report", "", f"Total plays: {len(segments)}", ""]

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
