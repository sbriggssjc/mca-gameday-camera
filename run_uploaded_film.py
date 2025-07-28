import sys
import os
from datetime import datetime
import argparse
from pathlib import Path

log_dir = "/logs/pipeline"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(log_dir, f"run_{timestamp}.log")
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

from manual_video_processor import process_uploaded_game_film


def main() -> None:
    parser = argparse.ArgumentParser(description="Process uploaded game film")
    parser.add_argument(
        "--video",
        required=True,
        help="Video file name within video/manual_uploads",
    )
    parser.add_argument(
        "--purge_after",
        action="store_true",
        help="Delete local video after successful upload",
    )
    parser.add_argument(
        "--max_frames_per_play",
        type=int,
        default=2,
        help="Maximum training frames to save per play",
    )
    parser.add_argument(
        "--prepare_retrain",
        action="store_true",
        help="Create retraining bundle after processing",
    )
    args = parser.parse_args()
    video_path = Path("video/manual_uploads") / args.video
    process_uploaded_game_film(
        str(video_path),
        purge_after=args.purge_after,
        max_frames_per_play=args.max_frames_per_play,
        prepare_retrain=args.prepare_retrain,
    )


if __name__ == "__main__":
    main()
    sys.stdout.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print(f"[✅] Log saved at: {log_path}")
