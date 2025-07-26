from __future__ import annotations
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class PlaybookEntry:
    """Single playbook entry with key attributes."""

    name: str
    formation: str
    play_type: str
    direction: str


@dataclass
class PlayResult:
    """Result of play recognition for one video segment."""

    name: str
    confidence: float
    frame_index: int
    formation: str
    direction: str
    play_type: str
    yardage: int | None
    outcome: str | None


def load_playbook(path: str) -> List[PlaybookEntry]:
    """Load playbook JSON as a list of :class:`PlaybookEntry`."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries: List[PlaybookEntry] = []
    if isinstance(data, dict):
        items = data.items()
    else:
        items = [(p.get("name", "unknown"), p) for p in data]

    for name, info in items:
        entries.append(
            PlaybookEntry(
                name=name,
                formation=str(info.get("formation", "")).lower(),
                play_type=str(info.get("type", "")).lower(),
                direction=str(info.get("direction", "")).lower(),
            )
        )
    return entries


def detect_play_attributes(frames: List[np.ndarray]) -> Tuple[str, str, str]:
    """Detect formation, direction and play type from frames.

    This is a heuristic placeholder using optical flow for direction.
    """
    if not frames:
        return "unknown", "middle", "run"

    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    flow_sum = 0.0
    count = 0
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_sum += float(flow[..., 0].mean())
        prev = gray
        count += 1
    mean_flow = flow_sum / max(1, count)
    if mean_flow > 0.2:
        direction = "right"
    elif mean_flow < -0.2:
        direction = "left"
    else:
        direction = "middle"
    # Placeholder formation and play type
    formation = "rit"
    play_type = "run"
    return formation, direction, play_type


def match_play(
    formation: str, direction: str, play_type: str, playbook: List[PlaybookEntry]
) -> Tuple[str, float]:
    """Return best matching play name and confidence score."""
    best_name = "unknown"
    best_score = -1.0
    for entry in playbook:
        score = 0.0
        if entry.formation and entry.formation == formation.lower():
            score += 1.0
        if entry.direction and entry.direction == direction.lower():
            score += 1.0
        if entry.play_type and entry.play_type == play_type.lower():
            score += 1.0
        confidence = score / 3.0
        if confidence > best_score:
            best_score = confidence
            best_name = entry.name
    return best_name, max(0.0, best_score)


def analyze_video(
    video_path: str, playbook: List[PlaybookEntry], output: str
) -> List[PlayResult]:
    """Analyze a video, returning a list of detected plays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    results: List[PlayResult] = []
    frames: List[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) >= 60:  # roughly every 2 seconds at 30 FPS
            formation, direction, play_type = detect_play_attributes(frames)
            name, conf = match_play(formation, direction, play_type, playbook)
            results.append(
                PlayResult(
                    name=name,
                    confidence=conf,
                    frame_index=frame_idx,
                    formation=formation,
                    direction=direction,
                    play_type=play_type,
                    yardage=None,
                    outcome=None,
                )
            )
            frames.clear()
        frame_idx += 1

    cap.release()
    write_results(results, output)
    return results


def write_results(results: List[PlayResult], path: str) -> None:
    """Append play recognition results to ``path`` in CSV or JSON format."""
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
                        "type",
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
                        r.play_type,
                        r.yardage if r.yardage is not None else "",
                        r.outcome if r.outcome is not None else "",
                    ]
                )
    else:
        data = [vars(r) for r in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def summarize_results(results: List[PlayResult]) -> Dict[str, int]:
    """Return a simple frequency count of plays."""
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.name] = counts.get(r.name, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze plays in a video")
    parser.add_argument("video", help="Path to game footage")
    parser.add_argument(
        "--playbook",
        default="mca_full_playbook_final.json",
        help="Playbook JSON",
    )
    parser.add_argument("--output", default="play_log.json", help="Output JSON/CSV")
    args = parser.parse_args()

    playbook = load_playbook(args.playbook)
    results = analyze_video(args.video, playbook, args.output)

    summary = summarize_results(results)
    for name, cnt in summary.items():
        print(f"{name}: {cnt}")


if __name__ == "__main__":
    main()
