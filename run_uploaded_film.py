import argparse
from pathlib import Path
from manual_video_processor import process_uploaded_game_film


def main() -> None:
    parser = argparse.ArgumentParser(description="Process uploaded game film")
    parser.add_argument(
        "--video",
        required=True,
        help="Video file name within video/manual_uploads",
    )
    args = parser.parse_args()
    video_path = Path("video/manual_uploads") / args.video
    process_uploaded_game_film(str(video_path))


if __name__ == "__main__":
    main()
