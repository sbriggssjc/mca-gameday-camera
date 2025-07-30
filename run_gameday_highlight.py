from __future__ import annotations

import argparse
from pathlib import Path

from generate_highlights import generate
from ai_tracking import analyze_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate game highlights with optional player tracking"
    )
    parser.add_argument("--input", required=True, help="Path to raw game video")
    parser.add_argument(
        "--track",
        action="store_true",
        help="Run player tracking analysis before generating highlights",
    )
    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if args.track:
        analyze_video(str(video_path))

    generate(str(video_path))


if __name__ == "__main__":
    main()
