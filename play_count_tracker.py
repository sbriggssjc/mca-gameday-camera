"""Manual play count tracker with periodic email summaries and alerts."""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, Iterable, List

import schedule
from colorama import Fore, Style, init as colorama_init

from email_alerts import load_env, send_email
from ai_detector import detect_jerseys
from google_sheets_uploader import format_row, upload_rows
import roster
from twilio.rest import Client
import pyttsx3

# Minimum plays before an alert is raised
ALERT_THRESHOLD = 7
# Minutes per quarter when --quarters mode is enabled
QUARTER_LENGTH = 8

JERSEY_NUMBERS: List[int] = sorted(roster.ROSTER.keys())
COUNTS_PATH = "jersey_counts.csv"


def summary_lines(counts: Dict[int, int], *, color: bool = False) -> List[str]:
    """Return formatted play count lines with player names."""
    lines = []
    for num in JERSEY_NUMBERS:
        cnt = counts.get(num, 0)
        alert = " < 7!" if cnt < ALERT_THRESHOLD else ""
        name = roster.get_player_name(num)
        text = f"#{num} {name}: {cnt}{alert}"
        if color and cnt < ALERT_THRESHOLD:
            text = f"{Fore.RED}{text}{Style.RESET_ALL}"
        lines.append(text)
    return lines


def email_summary(counts: Dict[int, int]) -> None:
    env = load_env()
    recipients = [e for e in env.get("COACH_EMAILS", "").split(',') if e]
    if not recipients:
        print("No coach email addresses configured; skipping email.")
        return
    subject = f"Play Count Update {time.strftime('%H:%M')}"
    body = "\n".join(summary_lines(counts))
    print("\n".join(["\nEmail Summary:"] + summary_lines(counts)))
    send_email(subject, body, recipients)


def send_sms_alert(message: str, env: Dict[str, str]) -> None:
    """Send a text message alert using Twilio if credentials are configured."""
    sid = env.get("TWILIO_SID")
    token = env.get("TWILIO_TOKEN")
    from_num = env.get("TWILIO_FROM")
    recipients = [p for p in env.get("COACH_PHONES", "").split(',') if p]
    if not (sid and token and from_num and recipients):
        return
    try:
        client = Client(sid, token)
        for phone in recipients:
            client.messages.create(body=message, from_=from_num, to=phone)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"Failed to send SMS: {exc}")


def manual_input_loop(
    counts: Dict[int, int],
    *,
    alerts: bool,
    voice: bool,
    env: Dict[str, str],
    quarter_counts: Dict[int, Dict[int, int]],
    current_quarter: List[int],
) -> None:
    play_number = 1
    engine = pyttsx3.init() if voice else None
    with open(COUNTS_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["play_number", "jerseys"])
        while True:
            schedule.run_pending()
            user_input = input(f"Players for play {play_number} (or q): ")
            if user_input.lower() in {"q", "quit"}:
                break
            try:
                nums = [int(x) for x in user_input.split()]
            except ValueError:
                print("Invalid input. Use jersey numbers separated by spaces.")
                continue
            invalid = [n for n in nums if n not in JERSEY_NUMBERS]
            if invalid:
                print(f"Invalid jersey numbers: {invalid}")
                continue
            writer.writerow([play_number, " ".join(str(n) for n in nums)])
            for n in set(nums):
                counts[n] = counts.get(n, 0) + 1
                q = current_quarter[0]
                quarter_counts[q][n] = quarter_counts[q].get(n, 0) + 1
                if alerts and counts[n] < ALERT_THRESHOLD:
                    msg = f"Player #{n} {roster.get_player_name(n)} below play count!"
                    print(Fore.RED + msg + Style.RESET_ALL)
                    if voice and engine:
                        engine.say(msg)
                        engine.runAndWait()
                    send_sms_alert(msg, env)
            play_number += 1


def ai_loop(
    counts: Dict[int, int],
    *,
    alerts: bool,
    voice: bool,
    env: Dict[str, str],
    quarter_counts: Dict[int, Dict[int, int]],
    current_quarter: List[int],
) -> None:
    print("AI mode stub running. Press Ctrl+C to stop.")
    engine = pyttsx3.init() if voice else None
    try:
        while True:
            detected = detect_jerseys()
            for n in set(detected):
                if n in JERSEY_NUMBERS:
                    counts[n] = counts.get(n, 0) + 1
                    q = current_quarter[0]
                    quarter_counts[q][n] = quarter_counts[q].get(n, 0) + 1
                    if alerts and counts[n] < ALERT_THRESHOLD:
                        msg = f"Player #{n} {roster.get_player_name(n)} below play count!"
                        print(Fore.RED + msg + Style.RESET_ALL)
                        if voice and engine:
                            engine.say(msg)
                            engine.runAndWait()
                        send_sms_alert(msg, env)
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track play counts with alerts")
    parser.add_argument("--ai", action="store_true", help="Use AI jersey detection")
    parser.add_argument("--no-alerts", action="store_true", help="Disable alerts")
    parser.add_argument("--voice", action="store_true", help="Enable voice alerts")
    parser.add_argument("--quarters", action="store_true", help="Use quarter timers")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    colorama_init()
    env = load_env()

    counts: Dict[int, int] = {n: 0 for n in JERSEY_NUMBERS}
    quarter_counts: Dict[int, Dict[int, int]] = {1: {}, 2: {}, 3: {}, 4: {}}
    current_quarter = [1]

    def quarter_job() -> None:
        q = current_quarter[0]
        print(f"\nQuarter {q} summary:")
        for line in summary_lines(counts, color=True):
            print(line)
        email_summary(counts)
        msg = f"Quarter {q} summary sent"
        send_sms_alert(msg, env)
        rows = [format_row(j, counts[j], [quarter_counts[x].get(j, 0) for x in range(1, 5)]) for j in JERSEY_NUMBERS]
        if env.get("SHEETS_CREDENTIALS") and env.get("SPREADSHEET_ID"):
            upload_rows(env["SHEETS_CREDENTIALS"], env["SPREADSHEET_ID"], rows)
        if q < 4:
            current_quarter[0] += 1

    if args.quarters:
        schedule.every(QUARTER_LENGTH).minutes.do(quarter_job)
    else:
        schedule.every(20).minutes.do(lambda: email_summary(counts))

    if args.test:
        for _ in range(20):
            for j in JERSEY_NUMBERS:
                counts[j] = counts.get(j, 0) + 1
            schedule.run_pending()
            time.sleep(0.1)
        print("Test mode complete")
    elif args.ai:
        ai_loop(
            counts,
            alerts=not args.no_alerts,
            voice=args.voice,
            env=env,
            quarter_counts=quarter_counts,
            current_quarter=current_quarter,
        )
    else:
        manual_input_loop(
            counts,
            alerts=not args.no_alerts,
            voice=args.voice,
            env=env,
            quarter_counts=quarter_counts,
            current_quarter=current_quarter,
        )

    print("\nFinal Counts:")
    for line in summary_lines(counts, color=True):
        print(line)
    email_summary(counts)


if __name__ == "__main__":
    main()
