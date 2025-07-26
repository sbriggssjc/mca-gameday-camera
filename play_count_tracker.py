"""Manual play count tracker with periodic email summaries."""

from __future__ import annotations

import csv
import os
import schedule
import time
from typing import Dict, Iterable, List

from email_alerts import load_env, send_email

JERSEY_NUMBERS: List[int] = [2, 3, 5, 7, 8, 9, 10, 11, 12, 14, 15, 17, 20, 21, 22, 24, 25]
COUNTS_PATH = "jersey_counts.csv"


def summary_lines(counts: Dict[int, int]) -> List[str]:
    lines = []
    for num in JERSEY_NUMBERS:
        cnt = counts.get(num, 0)
        alert = " < 7!" if cnt < 7 else ""
        lines.append(f"#{num}: {cnt}{alert}")
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


def manual_input_loop(counts: Dict[int, int]) -> None:
    play_number = 1
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
            play_number += 1


def detect_jerseys_stub() -> List[int]:
    """Placeholder for future AI-based detection."""
    return []


def ai_loop(counts: Dict[int, int]) -> None:
    print("AI mode stub running. Press Ctrl+C to stop.")
    try:
        while True:
            detected = detect_jerseys_stub()
            for n in set(detected):
                if n in JERSEY_NUMBERS:
                    counts[n] = counts.get(n, 0) + 1
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main() -> None:
    counts: Dict[int, int] = {n: 0 for n in JERSEY_NUMBERS}
    schedule.every(20).minutes.do(email_summary, counts)

    mode = os.environ.get("PLAY_TRACKER_MODE", "manual")
    if mode == "ai":
        ai_loop(counts)
    else:
        manual_input_loop(counts)

    print("\nFinal Counts:")
    for line in summary_lines(counts):
        print(line)
    email_summary(counts)


if __name__ == "__main__":
    main()
