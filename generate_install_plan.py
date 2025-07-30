import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

try:
    from fpdf import FPDF  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FPDF = None  # type: ignore


Row = Dict[str, str]


def load_rows(csv_path: Path, opponent: str) -> List[Row]:
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


def average_dict(items: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: sum(v) / len(v) if v else 0.0 for k, v in items.items()}


def top_plays(rows: List[Row], offense: str, n: int = 6) -> List[str]:
    plays: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r.get("offense", "").lower() != offense.lower():
            continue
        play = r.get("label", "unknown").strip() or "unknown"
        try:
            y = float(r.get("yards_gained", "0"))
        except Exception:
            y = 0.0
        plays[play].append(y)
    averages = average_dict(plays)
    ordered = sorted(averages.items(), key=lambda x: x[1], reverse=True)
    return [f"{p} (Avg: {a:.1f} yds)" for p, a in ordered[:n]]


def plays_allowed(rows: List[Row], offense: str, n: int = 6) -> List[str]:
    plays: Dict[str, List[float]] = defaultdict(list)
    formations: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r.get("offense", "").lower() == offense.lower():
            continue
        play = r.get("label", "unknown").strip() or "unknown"
        form = r.get("formation", "unknown").strip() or "unknown"
        try:
            y = float(r.get("yards_gained", "0"))
        except Exception:
            y = 0.0
        plays[play].append(y)
        formations[form].append(y)
    play_avgs = average_dict(plays)
    form_avgs = average_dict(formations)
    top_plays = [f"{p} (Avg: {a:.1f} yds)" for p, a in sorted(play_avgs.items(), key=lambda x: x[1], reverse=True)[:n]]
    weak_forms = [f"{f} (Avg: {a:.1f} yds)" for f, a in sorted(form_avgs.items(), key=lambda x: x[1], reverse=True)[:3]]
    return top_plays + weak_forms


def load_patterns(opponent: str) -> List[str]:
    safe = opponent.strip().replace(" ", "_")
    path = Path("analysis") / f"{safe}_scouting_report.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    lines: List[str] = []
    for p in data.get("patterns", []):
        form = p.get("formation", "")
        play = p.get("play", "")
        q = p.get("quarter")
        rate = p.get("rate")
        if form or play:
            desc = f"{form} {play}".strip()
            if q:
                desc += f" Q{q}"
            if rate:
                desc += f" {rate:.0%}"
            lines.append(desc)
    return lines


def read_notes(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return lines[:2]


def save_plan(title: str, offense: List[str], defense: List[str], schedule: List[str], out_path: Path) -> None:
    if FPDF is None:
        text_lines = [title, "", "Offensive Focus:"] + [f"- {o}" for o in offense]
        text_lines += ["", "Defensive Emphasis:"] + [f"- {d}" for d in defense]
        text_lines += ["", "Install Schedule:"] + [f"- {s}" for s in schedule]
        out_path.write_text("\n".join(text_lines) + "\n")
        return

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=1)
    pdf.set_font("Helvetica", size=12)

    pdf.cell(0, 10, "Offensive Focus:", ln=1)
    for o in offense:
        pdf.multi_cell(0, 8, f"- {o}")
    pdf.cell(0, 8, "", ln=1)

    pdf.cell(0, 10, "Defensive Emphasis:", ln=1)
    for d in defense:
        pdf.multi_cell(0, 8, f"- {d}")
    pdf.cell(0, 8, "", ln=1)

    pdf.cell(0, 10, "Install Schedule:", ln=1)
    for s in schedule:
        pdf.multi_cell(0, 8, f"- {s}")

    pdf.output(str(out_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weekly install plan")
    parser.add_argument("opponent", help="Opponent name")
    parser.add_argument("--week", type=int, default=1, help="Week number")
    parser.add_argument("--csv", default="scouting_data.csv", help="Scouting CSV data")
    parser.add_argument("--notes", default="coach_notes.txt", help="Optional coach notes file")
    parser.add_argument("--offense-name", default="MCA", help="Team offense identifier in CSV")
    parser.add_argument("--output-dir", default="install_plan", help="Directory for output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    rows = load_rows(csv_path, args.opponent)
    offense_lines = top_plays(rows, args.offense_name)
    defense_lines = plays_allowed(rows, args.offense_name)
    defense_lines.extend(load_patterns(args.opponent))
    defense_lines.extend(read_notes(Path(args.notes)))

    schedule = [
        "Monday: Install core plays and counters",
        "Tuesday: Add play action and defensive looks",
        "Thursday: Full script and red zone"
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_opp = args.opponent.strip().replace(" ", "_")
    suffix = "pdf" if FPDF is not None else "txt"
    out_path = out_dir / f"Week_{args.week}_{safe_opp}.{suffix}"
    title = f"Weekly Install Plan – Week {args.week} – vs {args.opponent}"
    save_plan(title, offense_lines, defense_lines, schedule, out_path)
    print(f"Plan saved to {out_path}")


if __name__ == "__main__":
    main()

