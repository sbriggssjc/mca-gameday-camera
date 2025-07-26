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


def detect_play_attributes(frames: List[np.ndarray]) -> Tuple[str, str, str, float]:
    """Detect formation, direction and play type from frames.

    This is a heuristic placeholder using optical flow for direction. The
    returned ``mean_flow`` represents the average horizontal optical flow and
    can be used to compare clips for training or calibration purposes.
    """
    if not frames:
        return "unknown", "middle", "run", 0.0

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
    return formation, direction, play_type, mean_flow


def match_play(
    formation: str,
    direction: str,
    play_type: str,
    mean_flow: float,
    playbook: List[PlaybookEntry],
    training_data: Dict[str, Dict[str, float]] | None = None,
) -> Tuple[str, float]:
    """Return best matching play name and confidence score.

    Training data can optionally provide a ``mean_flow`` value for each play
    which is used to slightly bias the matching confidence toward similar
    movement patterns.
    """
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

        if training_data and entry.name in training_data:
            ref = training_data[entry.name]
            ref_flow = float(ref.get("mean_flow", 0.0))
            diff = abs(mean_flow - ref_flow)
            flow_score = max(0.0, 1.0 - diff)
            # combine with base confidence, weighting flow_score moderately
            confidence = 0.7 * confidence + 0.3 * flow_score

        if confidence > best_score:
            best_score = confidence
            best_name = entry.name

    return best_name, max(0.0, best_score)


def analyze_video(
    video_path: str,
    playbook: List[PlaybookEntry],
    output: str,
    training_data: Dict[str, Dict[str, float]] | None = None,
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
            formation, direction, play_type, mean_flow = detect_play_attributes(frames)
            name, conf = match_play(
                formation,
                direction,
                play_type,
                mean_flow,
                playbook,
                training_data,
            )
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


def load_training_data(path: str) -> Dict[str, Dict[str, float]]:
    """Load practice training data if available."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {
                str(d.get("play_name", "")): d
                for d in data
                if isinstance(d, dict) and "play_name" in d
            }
        elif isinstance(data, dict):
            return {str(k): v for k, v in data.items()}
    except Exception:
        return {}
    return {}


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
    parser.add_argument(
        "--training-data",
        default="training_set.json",
        help="Optional practice training data JSON",
    )
    args = parser.parse_args()

    playbook = load_playbook(args.playbook)
    training_data = load_training_data(args.training_data)
    results = analyze_video(args.video, playbook, args.output, training_data)

    summary = summarize_results(results)
    for name, cnt in summary.items():
        print(f"{name}: {cnt}")


if __name__ == "__main__":
    main()
