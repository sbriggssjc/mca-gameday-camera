"""CLI entry point for training and manual labeling workflows."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REVIEW_DIR = Path("manual_review")
LABELS_CSV = REVIEW_DIR / "labels.csv"


def run_training() -> None:
    """Invoke ``train_model.py`` to train on labeled clips."""
    subprocess.run([sys.executable, "train_model.py"], check=False)


def label_clips() -> None:
    """Prompt the user to label clips located in ``manual_review/``.

    Already labeled clips listed in ``labels.csv`` are skipped. New labels are
    appended to the CSV file with columns ``filepath`` and ``label``.
    """
    REVIEW_DIR.mkdir(exist_ok=True)

    labeled: set[str] = set()
    if LABELS_CSV.exists():
        with LABELS_CSV.open(newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    labeled.add(row[0])

    clips = [p for p in REVIEW_DIR.glob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi"}]
    clips = [p for p in clips if str(p) not in labeled]

    if not clips:
        print("No untagged clips found.")
        return

    need_header = not LABELS_CSV.exists()
    with LABELS_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(["filepath", "label"])
        for clip in clips:
            label = input(f"Label for {clip.name} (Run/Pass/Sweep/Kickoff/etc): ").strip()
            if not label:
                label = "unknown"
            writer.writerow([clip.as_posix(), label])
            print(f"Recorded {clip.name} -> {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI training and labeling workflows")
    parser.add_argument("--train", action="store_true", help="Train model on labeled clips")
    parser.add_argument("--label", action="store_true", help="Manually label clips in manual_review/")
    args = parser.parse_args()

    ran = False
    if args.label:
        label_clips()
        ran = True
    if args.train:
        run_training()
        ran = True
    if not ran:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
