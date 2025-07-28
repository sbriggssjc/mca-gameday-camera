from __future__ import annotations

import sys
import os
from datetime import datetime
from pathlib import Path
import argparse

log_dir = "./logs/pipeline"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(log_dir, f"run_{timestamp}.log")
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

from gdrive_utils import upload_to_google_drive
from manual_video_processor import process_uploaded_game_film


UPLOAD_DIR = Path("video/manual_uploads")
SUMMARY_DIR = Path("output/summary")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process all uploaded videos")
    parser.add_argument(
        "--prepare_retrain",
        action="store_true",
        help="Create retraining bundle after processing each video",
    )
    args = parser.parse_args()

    videos = [p for p in UPLOAD_DIR.iterdir() if p.suffix.lower() in {".mp4", ".mov"}]
    print(f"Found {len(videos)} video(s) to process\n")

    uploaded: list[str] = []
    retained: list[str] = []

    for video in videos:
        print(f"Processing {video.name}...")
        try:
            process_uploaded_game_film(
                str(video), purge_after=False, prepare_retrain=args.prepare_retrain
            )
        except Exception as exc:
            print(f"⚠️ Failed to process {video.name}: {exc}")
            retained.append(video.name)
            continue

        summary_path = SUMMARY_DIR / f"{video.stem}_summary.json"
        video_ok = upload_to_google_drive(str(video), "GameFilmUploads")
        summary_ok = upload_to_google_drive(str(summary_path), "GameFilmSummaries")

        if video_ok and summary_ok:
            try:
                os.remove(video)
                uploaded.append(video.name)
                print(f"✅ {video.name} processed and uploaded successfully\n")
            except Exception as exc:
                retained.append(video.name)
                print(f"⚠️ {video.name} uploaded but could not be deleted: {exc}\n")
        else:
            retained.append(video.name)
            print(f"⚠️ {video.name} processed but upload failed — local file retained\n")

    print("Summary:\n------")
    print(f"Total files found: {len(videos)}")
    print(f"Files successfully uploaded and purged: {len(uploaded)}")
    for name in uploaded:
        print(f"  - {name}")
    if retained:
        print(f"Files retained due to errors: {len(retained)}")
        for name in retained:
            print(f"  - {name}")
    else:
        print("No files were retained")


if __name__ == "__main__":
    main()
    sys.stdout.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print(f"[✅] Log saved at: {log_path}")
