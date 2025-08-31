from __future__ import annotations

<<<<<<< HEAD
<<<<<<< HEAD
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

=======
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
=======
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

>>>>>>> 3fb8c6c8bd1feab7561579284c161798bd1142cb
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
<<<<<<< HEAD
<<<<<<< HEAD
        logging.info(f"\u2705 Saved predictions to {output_dir / 'predictions.json'}")
=======
        print(f"\u2705 Saved predictions to {output_dir / 'predictions.json'}")


>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
=======
        logging.info(f"\u2705 Saved predictions to {output_dir / 'predictions.json'}")
>>>>>>> 3fb8c6c8bd1feab7561579284c161798bd1142cb
if __name__ == "__main__":
    main()
