"""Training harness for self-learning using sample footage.

This script extracts frames from an input video, applies the existing
play recognition heuristics, optionally prompts the user to correct
predictions, and stores the results for later model tuning.

It can also evaluate prediction accuracy against a previously labeled
video to estimate precision and recall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from play_recognizer import load_playbook, detect_play_attributes, match_play


def extract_clips(video_path: str, clip_len: int = 60) -> List[List[cv2.Mat]]:
    """Return a list of frame sequences of length ``clip_len`` from ``video_path``."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    clips: List[List[cv2.Mat]] = []
    frames: List[cv2.Mat] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) == clip_len:
            clips.append(frames)
            frames = []
    cap.release()
    if frames:
        clips.append(frames)
    return clips


def auto_label(clips: List[List[cv2.Mat]], playbook) -> List[Tuple[str, float, cv2.Mat]]:
    """Generate play predictions and return ``(name, confidence, frame)`` per clip."""
    predictions: List[Tuple[str, float, cv2.Mat]] = []
    for clip in clips:
        formation, direction, play_type, mean_flow = detect_play_attributes(clip)
        name, conf = match_play(formation, direction, play_type, mean_flow, playbook)
        frame = clip[min(9, len(clip) - 1)]  # representative frame
        predictions.append((name, conf, frame))
    return predictions


def interactive_review(preds: List[Tuple[str, float, cv2.Mat]]) -> List[Dict[str, object]]:
    """Prompt user to accept or correct predictions."""
    labeled: List[Dict[str, object]] = []
    for idx, (name, conf, _frame) in enumerate(preds):
        prompt = f"Clip {idx}: predicted '{name}' ({conf:.2f}). Enter label or leave blank to accept: "
        user = input(prompt).strip()
        label = user or name
        labeled.append({"clip": idx, "play": label, "confidence": conf})
    return labeled


def save_results(labels: List[Dict[str, object]], label_path: Path) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


def evaluate(video_path: str, label_path: Path, playbook) -> Tuple[float, float]:
    """Return precision and recall comparing predictions against ``label_path``."""
    with open(label_path, "r", encoding="utf-8") as f:
        truth = {int(item["clip"]): item["play"] for item in json.load(f)}

    clips = extract_clips(video_path)
    preds = auto_label(clips, playbook)

    correct = 0
    for idx, (name, _conf, _frame) in enumerate(preds):
        if truth.get(idx) == name:
            correct += 1
    precision = correct / max(1, len(preds))
    recall = correct / max(1, len(truth))
    return precision, recall


def main() -> None:
    parser = argparse.ArgumentParser(description="AI training harness")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--dataset", default="training/dataset", help="Dataset root directory")
    parser.add_argument("--label", action="store_true", help="Review and correct predictions")
    parser.add_argument("--train", action="store_true", help="Store labeled results")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate against existing labels")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    frames_dir = dataset / "frames"
    labels_dir = dataset / "labels"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    playbook_file = (
        "mca_full_playbook_final.json"
        if Path("mca_full_playbook_final.json").exists()
        else "mca_playbook.json"
    )
    playbook = load_playbook(playbook_file)

    if args.evaluate:
        label_path = labels_dir / f"{Path(args.video).stem}.json"
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        precision, recall = evaluate(args.video, label_path, playbook)
        print(f"precision {precision:.2%} recall {recall:.2%}")
        return

    clips = extract_clips(args.video)
    preds = auto_label(clips, playbook)

    labels: List[Dict[str, object]]
    if args.label:
        labels = interactive_review(preds)
    else:
        labels = [
            {"clip": i, "play": name, "confidence": conf}
            for i, (name, conf, _frame) in enumerate(preds)
        ]

    if args.train or args.label:
        label_path = labels_dir / f"{Path(args.video).stem}.json"
        save_results(labels, label_path)
        for idx, (_name, _conf, frame) in enumerate(preds):
            frame_path = frames_dir / f"{Path(args.video).stem}_clip{idx}.jpg"
            cv2.imwrite(str(frame_path), frame)
        print(f"✅ Labels saved to {label_path}")
    else:
        for i, (name, conf, _frame) in enumerate(preds):
            print(f"clip {i}: {name} ({conf:.2f})")


if __name__ == "__main__":
    main()
