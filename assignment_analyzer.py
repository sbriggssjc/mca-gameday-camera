"""Placeholder player rating engine using jersey detection."""

from __future__ import annotations

import argparse
import csv
import json
import os
<<<<<<< HEAD
from typing import Dict, List, Tuple

import cv2
import logging

from ai_detector import detect_jerseys as _detect_jerseys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
Box = Tuple[int, int, int, int]
=======
from typing import Dict, List

import cv2

from ai_detector import detect_jerseys
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa


CSV_HEADER = ["play_id", "jersey", "assignment", "timestamp"]


def analyze_clip(video_path: str, assignments: Dict[int, str] | None = None, *, output: str = "player_ratings.csv") -> None:
    """Process a video clip and append results to ``output``.

    The current implementation calls :func:`ai_detector.detect_jerseys` on each
    frame and records which jerseys were seen at what time. Real movement
    analysis and rating logic is not implemented because the required models and
    playbook data are unavailable in this environment.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
<<<<<<< HEAD
        logging.error("Unable to open %s", video_path)
        return

    ret, frame = cap.read()
    if not ret or frame is None:
        logging.error("Failed to read first frame from %s", video_path)
        cap.release()
=======
        print(f"Unable to open {video_path}")
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
        return

    new_file = not os.path.exists(output)
    with open(output, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if new_file:
            writer.writerow(CSV_HEADER)

<<<<<<< HEAD
        while ret and frame is not None:
            boxes: List[Box] = detect_players(frame)
            jerseys: List[int] = detect_jerseys(frame, boxes)
=======
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            jerseys: List[int] = detect_jerseys(frame)
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            for num in jerseys:
                assignment = assignments.get(num, "unknown") if assignments else "unknown"
                writer.writerow([os.path.basename(video_path), num, assignment, timestamp])
<<<<<<< HEAD
            ret, frame = cap.read()
=======
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa

    cap.release()


<<<<<<< HEAD
# --- Detection helpers ---


def detect_players(frame) -> List[Box]:
    """Return list of person bounding boxes (x1,y1,x2,y2). Stub if no model yet."""
    try:
        # TODO: replace with your real detector/tracker; for now a safe stub
        return []
    except Exception as e:
        logging.warning("detect_players failed: %s", e)
        return []


def detect_jerseys(frame, boxes: List[Box]) -> List[int]:
    """Return jersey numbers detected within ``boxes`` on ``frame``.

    The previous implementation allowed ``boxes`` to be omitted which
    resulted in ``TypeError`` whenever :func:`ai_detector.detect_jerseys`
    was invoked without the required argument.  Enforce the explicit
    ``boxes`` parameter so call sites must provide the current detection
    boxes.  This mirrors the signature of
    :func:`ai_detector.detect_jerseys` and prevents silent mis-use.
    """

    return _detect_jerseys(frame, boxes)

=======
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
def load_assignments(path: str) -> Dict[int, str]:
    """Load a simple jersey->assignment mapping from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    try:
        return {int(k): str(v) for k, v in data.items()}
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rate players per play (stub)")
    parser.add_argument("video", help="Path to a video clip")
<<<<<<< HEAD
    parser.add_argument("--playbook", help="Path to playbook JSON", default="playbooks/mca_5th_playbook.json")
=======
    parser.add_argument("--playbook", help="Path to playbook JSON", default=None)
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
    parser.add_argument("--output", help="Output CSV file", default="player_ratings.csv")
    args = parser.parse_args()

    assignments = load_assignments(args.playbook) if args.playbook else {}
    analyze_clip(args.video, assignments, output=args.output)


if __name__ == "__main__":
    main()
