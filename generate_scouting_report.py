"""Generate scouting report for a specific opponent using labeled play data."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from fpdf import FPDF  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FPDF = None  # type: ignore


Row = Dict[str, str]


def load_rows(csv_path: Path, opponent: str) -> List[Row]:
    """Return all rows in ``csv_path`` matching ``opponent``."""
    rows: List[Row] = []
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("opponent", "").lower() != opponent.lower():
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"No data for opponent: {opponent}")
    return rows


def counter_percent(counter: Counter[str], total: int) -> List[str]:
    """Return list of formatted lines sorted by count desc."""
    lines: List[str] = []
    for item, cnt in counter.most_common():
        pct = (cnt / total) * 100 if total else 0
        lines.append(f"- {item}: {pct:.1f}%")
    return lines


def analyze(rows: Iterable[Row]) -> Dict[str, Counter[str]]:
    """Compute frequency counters for various metrics."""
    plays = Counter()
    forms = Counter()
    first = Counter()
    third = Counter()
    scoring = Counter()
    quarters = Counter()
    for r in rows:
        label = r.get("label", "unknown").strip() or "unknown"
        form = r.get("formation", "unknown").strip() or "unknown"
        down = r.get("down", "").strip().lower()
        yards = r.get("yards_gained", "0").strip()
        qtr = r.get("quarter", "").strip()

        plays[label] += 1
        forms[form] += 1
        if down in {"1", "1st", "first"}:  # allow variants
            first[label] += 1
        if down in {"3", "4", "3rd", "4th", "third", "fourth"}:
            third[label] += 1
        try:
            y = float(yards)
        except Exception:
            y = 0.0
        if y >= 10 or "td" in label.lower():
            scoring[label] += 1
        if qtr:
            quarters[qtr] += 1
    return {
        "plays": plays,
        "formations": forms,
        "first_down": first,
        "third_down": third,
        "scoring": scoring,
        "quarters": quarters,
    }


def generate_text(opponent: str, counts: Dict[str, Counter[str]]) -> str:
    total = sum(counts["plays"].values())
    lines = [f"Opponent Scouting Report: {opponent}", ""]

    lines.append("Most Used Plays:")
    lines.extend(counter_percent(counts["plays"], total)[:5])
    lines.append("")

    lines.append("Formation Usage:")
    lines.extend(counter_percent(counts["formations"], total))
    lines.append("")

    fd_total = sum(counts["first_down"].values())
    if fd_total:
        lines.append("Common 1st Down Calls:")
        lines.extend(counter_percent(counts["first_down"], fd_total)[:5])
        lines.append("")

    td_total = sum(counts["third_down"].values())
    if td_total:
        lines.append("Common 3rd/4th Down Calls:")
        lines.extend(counter_percent(counts["third_down"], td_total)[:5])
        lines.append("")

    sc_total = sum(counts["scoring"].values())
    if sc_total:
        lines.append("Scoring Plays:")
        lines.extend(counter_percent(counts["scoring"], sc_total)[:5])
        lines.append("")

    if counts["quarters"]:
        lines.append("Plays by Quarter:")
        q_total = sum(counts["quarters"].values())
        for q, cnt in sorted(counts["quarters"].items()):
            pct = (cnt / q_total) * 100 if q_total else 0
            lines.append(f"- {q}: {pct:.1f}%")
        lines.append("")

    return "\n".join(lines) + "\n"


def save_report(text: str, path: Path) -> None:
    """Save report text to ``path`` or PDF if fpdf is available."""
    if FPDF is None:
        path.write_text(text)
        return

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in text.splitlines():
        pdf.multi_cell(0, 10, line)
    pdf.output(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate opponent scouting report")
    parser.add_argument("opponent", help="Opponent name")
    parser.add_argument("--csv", default="scouting_data.csv", help="CSV data file")
    parser.add_argument(
        "--output-dir", default="analysis", help="Directory for the report"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    rows = load_rows(csv_path, args.opponent)
    counts = analyze(rows)
    text = generate_text(args.opponent, counts)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.opponent.strip().replace(" ", "_")
    suffix = "pdf" if FPDF is not None else "txt"
    out_path = out_dir / f"{safe_name}_scouting_report.{suffix}"
    save_report(text, out_path)
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()

