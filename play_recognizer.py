from __future__ import annotations
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


@dataclass
class PlayResult:
    name: str
    confidence: float
    frame_index: int
    formation: str
    direction: str
    yardage: int | None
    outcome: str | None


def load_playbook(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_formation_and_direction(_frames: List) -> Tuple[str, str]:
    """Placeholder detection of formation and direction."""
    # Real implementation would use YOLOv8 and player tracking.
    return "Rit", "right"


def match_play(
    formation: str, direction: str, playbook: List[Dict[str, str]]
) -> Tuple[str, float]:
    """Return best matching play name and confidence score."""
    best_name = "unknown"
    best_score = 0
    for play in playbook:
        score = 0
        if play.get("formation") == formation:
            score += 1
        if play.get("direction") == direction:
            score += 1
        if score > best_score:
            best_score = score
            best_name = play.get("name", "unknown")
    confidence = best_score / 2.0
    return best_name, confidence


def analyze_video(
    video_path: str, playbook: List[Dict[str, str]], output: str
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    results: List[PlayResult] = []
    frame_idx = 0
    frames: List = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) >= 60:  # analyze every ~2 seconds at 30 FPS
            formation, direction = detect_formation_and_direction(frames)
            name, confidence = match_play(formation, direction, playbook)
            results.append(
                PlayResult(
                    name=name,
                    confidence=confidence,
                    frame_index=frame_idx,
                    formation=formation,
                    direction=direction,
                    yardage=None,
                    outcome=None,
                )
            )
            frames.clear()
        frame_idx += 1

    cap.release()
    write_results(results, output)


def write_results(results: List[PlayResult], path: str) -> None:
    if path.endswith(".csv"):
        new_file = not Path(path).exists()
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(
                    [
                        "frame",
                        "play",
                        "confidence",
                        "formation",
                        "direction",
                        "yardage",
                        "outcome",
                    ]
                )
            for r in results:
                writer.writerow(
                    [
                        r.frame_index,
                        r.name,
                        f"{r.confidence:.2f}",
                        r.formation,
                        r.direction,
                        r.yardage if r.yardage is not None else "",
                        r.outcome if r.outcome is not None else "",
                    ]
                )
    else:
        data = [vars(r) for r in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)



def summarize_results(results: List[PlayResult]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.name] = counts.get(r.name, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze plays in a video")
    parser.add_argument("video", help="Path to game footage")
    parser.add_argument("--playbook", default="mca_playbook.json", help="Playbook JSON")
    parser.add_argument("--output", default="play_log.json", help="Output JSON/CSV")
    args = parser.parse_args()

    playbook = load_playbook(args.playbook)
    analyze_video(args.video, playbook, args.output)


if __name__ == "__main__":
    main()
