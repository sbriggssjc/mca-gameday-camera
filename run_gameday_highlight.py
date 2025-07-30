from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_highlights import generate
from ai_tracking import analyze_video
from play_classifier import classify_play


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
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Label highlight clips with play type predictions",
    )
    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if args.track:
        analyze_video(str(video_path))

    output_dir = Path("highlights")
    generate(str(video_path), str(output_dir))

    if args.classify:
        preds = []
        for clip in sorted(output_dir.glob("*.mp4")):
            meta = clip.with_suffix(".json")
            meta_path = str(meta) if meta.exists() else None
            result = classify_play(str(clip), meta_path)
            preds.append({
                "clip": clip.name,
                "play_type": result["play_type"],
                "confidence": result["confidence"],
            })
        with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2)
        print(f"\u2705 Saved predictions to {output_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
