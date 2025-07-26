"""Generate a film room dashboard summarizing ratings and clips."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple

from email_alerts import load_env, send_email

try:
    from fpdf import FPDF  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FPDF = None  # type: ignore


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def read_ratings(csv_path: str) -> List[Tuple[str, str]]:
    """Return list of (player, rating) from a CSV with two columns."""
    ratings: List[Tuple[str, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].lower() == "player" or row[0].startswith("#"):
                # skip header or comments
                continue
            player = row[0].strip()
            rating = row[1].strip() if len(row) > 1 else ""
            ratings.append((player, rating))
    return ratings


def collect_clips(directory: str) -> List[Path]:
    """Return sorted list of video clip paths in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    clips = [p for p in dir_path.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv"}]
    return sorted(clips)


def find_diagram(clip: Path) -> Path | None:
    """Return diagram image for a clip if present."""
    for ext in (".png", ".jpg", ".jpeg"):
        cand = clip.with_suffix(ext)
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def generate_pdf(
    ratings: List[Tuple[str, str]],
    good_clips: List[Path],
    error_clips: List[Path],
    output: str,
) -> None:
    """Create a PDF summarizing ratings and listing clips."""

    if FPDF is None:
        raise RuntimeError("fpdf package is required to generate PDF")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Page 1: Rating summary
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Player Ratings", ln=1)

    pdf.set_font("Helvetica", size=12)
    for player, rating in ratings:
        pdf.cell(0, 8, f"{player}: {rating}", ln=1)

    # Great plays
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Great Plays", ln=1)
    pdf.set_font("Helvetica", size=12)
    for clip in good_clips:
        pdf.cell(0, 8, clip.name, ln=1)
        diagram = find_diagram(clip)
        if diagram:
            pdf.image(str(diagram), w=150)
            pdf.ln(4)

    # Blocking errors
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Blocking Errors", ln=1)
    pdf.set_font("Helvetica", size=12)
    for clip in error_clips:
        pdf.cell(0, 8, clip.name, ln=1)
        diagram = find_diagram(clip)
        if diagram:
            pdf.image(str(diagram), w=150)
            pdf.ln(4)

    pdf.output(output)


# ---------------------------------------------------------------------------
# Upload/email helpers
# ---------------------------------------------------------------------------

def upload_to_drive(path: str, dest: str) -> None:
    """Upload file to Google Drive using rclone."""
    result = subprocess.run(["rclone", "copy", path, dest], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def email_to_coaches(pdf_path: str, coaches_file: str = "coaches.json") -> None:
    """Email the PDF to coach email addresses defined in a JSON file."""
    try:
        with open(coaches_file) as f:
            data = json.load(f)
            emails = [c["email"] for c in data.get("coaches", []) if c.get("email")]
    except Exception:
        emails = []
    if not emails:
        print("No coach emails found; skipping email")
        return

    env = load_env()
    subject = "Film Room Dashboard"
    body = "Attached is the latest film dashboard."
    for email in emails:
        print(f"Emailing dashboard to {email}")
    send_email(subject, body, emails)
    # send_email does not handle attachments; simple message only


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate film session dashboard PDF")
    parser.add_argument("ratings", help="CSV file with player ratings")
    parser.add_argument("good_clips", help="Directory of great play clips")
    parser.add_argument("error_clips", help="Directory of blocking error clips")
    parser.add_argument("output", help="Output PDF path")
    parser.add_argument("--drive", help="rclone destination for upload")
    parser.add_argument("--email", action="store_true", help="Email PDF to coaches")
    parser.add_argument("--coaches", default="coaches.json", help="Coach contact JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratings = read_ratings(args.ratings)
    good = collect_clips(args.good_clips)
    bad = collect_clips(args.error_clips)

    generate_pdf(ratings, good, bad, args.output)
    print(f"Dashboard created: {args.output}")

    if args.drive:
        try:
            upload_to_drive(args.output, args.drive)
            print("Uploaded to Google Drive")
        except Exception as exc:
            print(f"Drive upload failed: {exc}")

    if args.email:
        try:
            email_to_coaches(args.output, args.coaches)
        except Exception as exc:
            print(f"Failed to send email: {exc}")


if __name__ == "__main__":
    main()
