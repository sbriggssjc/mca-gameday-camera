#!/usr/bin/env python3
"""Simple smoke test for ball tracking pipeline."""

import argparse
import time
from typing import Union

import cv2
import numpy as np

from analysis.tracking.ball_tracker import BallTracker


def parse_source(src: str) -> Union[int, str]:
    """Return camera index if ``src`` is numeric otherwise the string."""
    try:
        return int(src)
    except ValueError:
        return src


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ball tracking pipeline for a short period")
    parser.add_argument("source", nargs="?", default="0", help="Video file or camera index")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    parser.add_argument(
        "--proc-scale",
        type=float,
        default=0.5,
        help="Downscale factor for processing",
    )
    args = parser.parse_args()

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    tracker = BallTracker(proc_scale=args.proc_scale)
    confidences = []
    frames = 0
    start = time.time()
    while time.time() - start < args.duration:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        result = tracker.update(frame)
        if result:
            # result is (x, y, w, h, conf, state)
            confidences.append(result[4])
    cap.release()

    assert frames > 0, "No frames recorded"

    elapsed = time.time() - start
    fps = frames / elapsed if elapsed > 0 else 0.0
    print(f"Processed {frames} frames in {elapsed:.2f}s ({fps:.2f} FPS)")

    if confidences:
        hist, bins = np.histogram(confidences, bins=10, range=(0.0, 1.0))
        print("Tracker confidence histogram:")
        for count, b0, b1 in zip(hist, bins[:-1], bins[1:]):
            print(f"{b0:.1f}-{b1:.1f}: {count}")
    else:
        print("No confidence values recorded")


if __name__ == "__main__":
    main()
