#!/usr/bin/env python3
"""Measure capture to display latency for a video source."""

import argparse
import time
from typing import Union

import cv2


def parse_source(src: str) -> Union[int, str]:
    try:
        return int(src)
    except ValueError:
        return src


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure capture to display latency")
    parser.add_argument("source", nargs="?", default="0", help="Video file or camera index")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to sample")
    args = parser.parse_args()

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    latencies = []
    for _ in range(args.frames):
        ret, frame = cap.read()
        if not ret:
            break
        capture_ts = time.time()
        # dummy processing step
        _ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        display_ts = time.time()
        latencies.append((display_ts - capture_ts) * 1000.0)
    cap.release()
    cv2.destroyAllWindows()

    assert latencies, "No frames processed"

    avg = sum(latencies) / len(latencies)
    maximum = max(latencies)
    print(f"Average latency: {avg:.1f} ms (max {maximum:.1f} ms)")
    if avg > 250 or maximum > 250:
        print("WARNING: Latency exceeds 250 ms target")


if __name__ == "__main__":
    main()
