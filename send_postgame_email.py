"""Send postgame summary emails to parents and coaches."""

from __future__ import annotations

import argparse
import csv
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, List

from email_alerts import load_env


def read_recipients(csv_path: str) -> List[str]:
    """Return list of email addresses from a CSV with Name,Email columns."""
    emails: List[str] = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0].lower() in {"name", "email"}:
                    # Skip header row
                    continue
                # Support rows like "Name, Email"
                email = row[1] if len(row) > 1 else row[0]
                email = email.strip()
                if email:
                    emails.append(email)
    except FileNotFoundError:
        raise SystemExit(f"Recipients file not found: {csv_path}")
    return emails


def build_body(
    opponent: str,
    youtube_link: str,
    drive_link: str,
    score_mca: str,
    score_opp: str,
    coach_note: str | None = None,
) -> str:
    lines = [
        "Hi MCA family,",
        "",
        f"Here's your postgame summary from our match vs {opponent}:",
        "",
        f"📺 Game Video: {youtube_link}",
        f"📂 Game Folder: {drive_link}",
        f"📊 Final Score: MCA {score_mca} – {score_opp}",
        "📌 Attached: Coach analysis summary (PDF)",
    ]
    if coach_note:
        lines.extend(["", "Coach's Note:", coach_note.strip()])
    lines.append("\nThank you for your support!")
    return "\n".join(lines)


def send_email(
    subject: str,
    body: str,
    recipients: Iterable[str],
    attachment: Path,
    *,
    dry_run: bool = False,
) -> None:
    env = load_env()
    smtp_user = env.get("SMTP_USER")
    smtp_pass = env.get("SMTP_PASS")
    if not smtp_user or not smtp_pass:
        if dry_run:
            smtp_user = smtp_user or "example@example.com"
            smtp_pass = smtp_pass or "unused"
        else:
            raise SystemExit("Missing SMTP_USER/SMTP_PASS environment variables")
    smtp_server = env.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(env.get("SMTP_PORT", "587"))

    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)

    with attachment.open("rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename=attachment.name,
        )

    if dry_run:
        print("--- DRY RUN ---")
        print("To:", msg["To"])
        print("Subject:", msg["Subject"])
        print("Body:\n", body)
        print(f"Attachment: {attachment}")
        return

    with smtplib.SMTP(smtp_server, smtp_port) as smtp:
        smtp.starttls()
        smtp.login(smtp_user, smtp_pass)
        smtp.send_message(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send postgame summary email")
    parser.add_argument("opponent", help="Opponent name")
    parser.add_argument("mca_score", help="MCA score")
    parser.add_argument("opp_score", help="Opponent score")
    parser.add_argument("youtube_link", help="YouTube video link")
    parser.add_argument("drive_link", help="Google Drive folder link")
    parser.add_argument(
        "--emails",
        default="emails.csv",
        help="CSV file with Name,Email columns",
    )
    parser.add_argument(
        "--notes",
        default="coach_notes.txt",
        help="Optional coach notes text file",
    )
    parser.add_argument(
        "--attachment",
        default="analysis/analysis_summary.pdf",
        help="PDF analysis file to attach",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview email instead of sending",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipients = read_recipients(args.emails)
    if not recipients:
        raise SystemExit("No recipients found")

    note = None
    if args.notes and os.path.exists(args.notes):
        note = Path(args.notes).read_text().strip()

    body = build_body(
        args.opponent,
        args.youtube_link,
        args.drive_link,
        args.mca_score,
        args.opp_score,
        note,
    )

    subject = f"🏈 MCA 5th Grade Postgame Summary – vs {args.opponent}"

    send_email(
        subject,
        body,
        recipients,
        Path(args.attachment),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
