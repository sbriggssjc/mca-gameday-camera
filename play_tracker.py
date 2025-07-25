"""Real-time play tracker that emails periodic summaries.

This script monitors jersey numbers from a (placeholder) real-time video feed
and tracks how many plays each player participates in. Every 20 minutes a play
count summary is emailed to the configured recipients.
"""

from __future__ import annotations

import argparse
import smtplib
import schedule
import time
from email.message import EmailMessage
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track play counts and send periodic email updates.")
    parser.add_argument(
        "--jerseys",
        "-j",
        nargs="+",
        type=int,
        help="List of jersey numbers to track",
    )
    parser.add_argument(
        "--config",
        help="Optional text file with jersey numbers (one per line)",
    )
    parser.add_argument(
        "--recipients",
        "-r",
        nargs="+",
        required=True,
        help="Email addresses to send updates to",
    )
    parser.add_argument("--smtp-user", required=True, help="SMTP username (Gmail address)")
    parser.add_argument("--smtp-pass", required=True, help="SMTP password or app password")
    parser.add_argument("--smtp-server", default="smtp.gmail.com", help="SMTP server hostname")
    parser.add_argument("--smtp-port", type=int, default=587, help="SMTP server port")
    return parser.parse_args()


def load_jerseys(args: argparse.Namespace) -> List[int]:
    jerseys: List[int] = []
    if args.jerseys:
        jerseys.extend(args.jerseys)
    if args.config:
        try:
            with open(args.config) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        jerseys.append(int(line))
        except Exception as exc:  # pragma: no cover - runtime failure is user error
            raise SystemExit(f"Failed to read config file: {exc}")
    if not jerseys:
        raise SystemExit("No jersey numbers provided")
    return sorted(set(jerseys))


def detect_jerseys(_frame=None) -> List[int]:
    """Placeholder jersey number detection."""
    # TODO: integrate actual detection logic using computer vision.
    return []


def send_email(
    subject: str,
    body: str,
    recipients: Iterable[str],
    *,
    smtp_user: str,
    smtp_pass: str,
    smtp_server: str,
    smtp_port: int,
) -> None:
    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(smtp_server, smtp_port) as smtp:
        smtp.starttls()
        smtp.login(smtp_user, smtp_pass)
        smtp.send_message(msg)


def main() -> None:
    args = parse_args()
    jerseys = load_jerseys(args)
    play_counts = {j: 0 for j in jerseys}
    quarter = 1

    def email_job() -> None:
        nonlocal quarter
        lines = [f"#{j} – {play_counts[j]} plays" for j in jerseys]
        body = "\n".join(lines)
        subject = f"Play Count Update – Q{quarter}"
        send_email(
            subject,
            body,
            args.recipients,
            smtp_user=args.smtp_user,
            smtp_pass=args.smtp_pass,
            smtp_server=args.smtp_server,
            smtp_port=args.smtp_port,
        )
        quarter += 1

    schedule.every(20).minutes.do(email_job)

    try:
        while True:
            detected = detect_jerseys()
            for num in set(detected):
                if num in play_counts:
                    play_counts[num] += 1
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
