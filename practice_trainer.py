import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from play_recognizer import load_playbook


def parse_play_name(filename: str) -> str:
    """Return play name from a clip filename."""
    name = Path(filename).stem
    name = re.sub(r"_Rep\d+$", "", name, flags=re.IGNORECASE)
    return name.replace("_", " ").strip().title()


def analyze_clip(path: Path) -> Dict[str, float]:
    """Compute simple motion features for a video clip."""
    cap = cv2.VideoCapture(str(path))
    success, frame = cap.read()
    if not success:
        cap.release()
        return {"mean_flow": 0.0, "mean_mag": 0.0, "path_consistency": 0.0, "direction": "middle"}
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flows: List[float] = []
    mags: List[float] = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(float(flow[..., 0].mean()))
        mags.append(float(np.linalg.norm(flow, axis=2).mean()))
        prev = gray
    cap.release()
    if flows:
        mean_flow = float(np.mean(flows))
        mean_mag = float(np.mean(mags))
        std_mag = float(np.std(mags))
    else:
        mean_flow = mean_mag = std_mag = 0.0
    if mean_flow > 0.2:
        direction = "right"
    elif mean_flow < -0.2:
        direction = "left"
    else:
        direction = "middle"
    # higher std -> lower consistency
    consistency = 1.0 - std_mag / (mean_mag + 1e-6)
    return {
        "mean_flow": mean_flow,
        "mean_mag": mean_mag,
        "path_consistency": consistency,
        "direction": direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Process practice clips for training")
    parser.add_argument("folder", help="Folder containing practice videos")
    parser.add_argument("--playbook", default="mca_full_playbook_final.json", help="Playbook JSON")
    parser.add_argument("--output", default="training_set.json", help="Output training JSON")
    args = parser.parse_args()

    playbook = {p.name: p for p in load_playbook(args.playbook)}

    data: List[Dict[str, float]] = []
    folder = Path(args.folder)
    for video in sorted(folder.glob("*.mp4")):
        play_name = parse_play_name(video.name)
        features = analyze_clip(video)
        entry = {
            "clip": video.name,
            "play_name": play_name,
            "mean_flow": features["mean_flow"],
            "mean_motion": features["mean_mag"],
            "path_consistency": features["path_consistency"],
            "direction": features["direction"],
        }
        pb = playbook.get(play_name)
        if pb is not None:
            entry["formation"] = pb.formation
            entry["play_type"] = pb.play_type
            if pb.direction and pb.direction.lower() != features["direction"].lower():
                print(f"Warning: {video.name} direction {features['direction']} != playbook {pb.direction}")
        else:
            print(f"Play {play_name} not in playbook")
        data.append(entry)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
