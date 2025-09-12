"""Generate a simple Markdown opponent report from tendency CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def _load_csv(path: Path) -> Dict[str, List[Dict[str, str]]]:
    if not path.exists():
        return {}
    out: Dict[str, List[Dict[str, str]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out.setdefault(row["metric"], []).append(row)
    return out


def _top(rows: List[Dict[str, str]], k: int = 3) -> List[Dict[str, str]]:
    rows.sort(key=lambda r: int(r.get("count", "0")), reverse=True)
    return rows[:k]


def _fmt_top(rows: List[Dict[str, str]]) -> List[str]:
    lines = []
    for r in rows:
        avg = float(r.get("avg_yards" or 0))
        lines.append(f"- {r['value']} ({r['count']} plays, avg {avg:.1f} yd)")
    return lines or ["- none"]


def build_section(title: str, csv_rows: Dict[str, List[Dict[str, str]]]) -> List[str]:
    lines = [f"## {title}"]
    rp = {r["value"]: r for r in csv_rows.get("run_pass", [])}
    if rp:
        run = rp.get("run", {"count": "0", "avg_yards": "0"})
        pas = rp.get("pass", {"count": "0", "avg_yards": "0"})
        lines.append(
            f"Run: {run['count']} (avg {float(run['avg_yards']):.1f} yd) | Pass: {pas['count']} (avg {float(pas['avg_yards']):.1f} yd)"
        )
    forms = _top(csv_rows.get("formation_text", []))
    if forms:
        lines.append("Top formations:")
        lines.extend(_fmt_top(forms))
    dirs = _top(csv_rows.get("run_direction", []))
    if dirs:
        lines.append("Run direction:")
        lines.extend(_fmt_top(dirs))
    routes = _top(csv_rows.get("route_primary", []))
    if routes:
        lines.append("Routes:")
        lines.extend(_fmt_top(routes))
    return lines


def main(out_dir: str) -> Path:
    out = Path(out_dir)
    plays = [json.loads(l) for l in (out / "plays.jsonl").read_text().splitlines() if l.strip()]
    total = len(plays)
    off_csv = _load_csv(out / "tendencies_offense.csv")
    def_csv = _load_csv(out / "tendencies_defense.csv")

    lines = [
        "# Opponent Report",
        f"**Total clips analysed:** {total}",
        "",
    ]
    lines.extend(build_section("Offense", off_csv))
    lines.append("")
    lines.extend(build_section("Defense", def_csv))

    report = out / "opponent_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m analysis.opponent_report <OUT>")
    main(sys.argv[1])

