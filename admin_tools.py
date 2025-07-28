"""Command line tools for administering training data and logs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import zipfile


TRAINING_DIR = Path("training")
LABELS_DIR = TRAINING_DIR / "labels"
LOGS_DIR = TRAINING_DIR / "logs"
BUNDLES_DIR = TRAINING_DIR / "bundles"
BACKUP_DIR = TRAINING_DIR / "backups"
FRAMES_DIR = TRAINING_DIR / "frames"
UNCERTAIN_DIR = TRAINING_DIR / "uncertain_jerseys"
REVIEW_QUEUE_PATH = TRAINING_DIR / "review_queue.json"


def zip_paths(zip_path: Path, paths: list[Path]) -> None:
    """Zip the given paths into ``zip_path``."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if not path.exists():
                continue
            if path.is_file():
                zf.write(path, path.relative_to(TRAINING_DIR))
            else:
                for file in path.rglob("*"):
                    if file.is_file():
                        zf.write(file, file.relative_to(TRAINING_DIR))


def export_training_bundle() -> None:
    """Export a bundle of training data as a zip archive."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = BUNDLES_DIR / f"training_bundle_{timestamp}.zip"
    zip_paths(zip_path, [FRAMES_DIR, LABELS_DIR, UNCERTAIN_DIR])
    print(f"Created {zip_path}")


def backup_logs() -> None:
    """Backup the training logs directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = BACKUP_DIR / f"logs_{timestamp}.zip"
    zip_paths(zip_path, [LOGS_DIR])
    print(f"Logs backed up to {zip_path}")


def clear_review_queue() -> None:
    """Truncate the review queue file."""
    REVIEW_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_QUEUE_PATH.write_text("[]\n")
    print("Review queue cleared")


def reset_labels() -> None:
    """Backup and truncate label files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for label_file in [LABELS_DIR / "confirmed_jerseys.json", LABELS_DIR / "confirmed_play_types.json"]:
        if label_file.exists():
            backup_path = BACKUP_DIR / f"{label_file.stem}_{timestamp}{label_file.suffix}"
            backup_path.write_bytes(label_file.read_bytes())
            label_file.write_text("[]\n")
            print(f"Backed up and cleared {label_file.name}")
        else:
            print(f"{label_file.name} not found")


def summary() -> None:
    """Print summary statistics about the training data."""
    jerseys_path = LABELS_DIR / "confirmed_jerseys.json"
    play_types_path = LABELS_DIR / "confirmed_play_types.json"

    def count_json(path: Path) -> int:
        if not path.exists():
            return 0
        try:
            return len(json.loads(path.read_text()))
        except json.JSONDecodeError:
            return 0

    jersey_count = count_json(jerseys_path)
    play_type_count = count_json(play_types_path)
    review_queue_len = count_json(REVIEW_QUEUE_PATH)
    unzipped = [p.name for p in BUNDLES_DIR.glob("*") if p.is_dir()]
    logs_count = len(list(LOGS_DIR.glob("*"))) if LOGS_DIR.exists() else 0

    print("Summary:")
    print(f"  Jerseys labeled: {jersey_count}")
    print(f"  Play types labeled: {play_type_count}")
    print(f"  Review queue items: {review_queue_len}")
    print(f"  Unzipped bundles: {len(unzipped)}")
    print(f"  Log files: {logs_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Admin tools for training data")
    parser.add_argument("--export_training_bundle", action="store_true", help="Export training bundle")
    parser.add_argument("--backup_logs", action="store_true", help="Backup logs directory")
    parser.add_argument("--clear_review_queue", action="store_true", help="Clear review queue file")
    parser.add_argument("--reset_labels", action="store_true", help="Backup and reset label files")
    parser.add_argument("--summary", action="store_true", help="Print summary of training data")
    args = parser.parse_args()

    if args.export_training_bundle:
        export_training_bundle()
    if args.backup_logs:
        backup_logs()
    if args.clear_review_queue:
        clear_review_queue()
    if args.reset_labels:
        reset_labels()
    if args.summary:
        summary()

    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - script
    main()
