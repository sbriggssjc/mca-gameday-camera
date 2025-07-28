from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import cv2

from ai_detector import detect_jerseys
from play_recognizer import (
    load_playbook,
    detect_play_attributes,
    match_play,
)
from scoreboard_reader import ScoreboardReader
from smart_auto_tracker import SmartAutoTracker
from gdrive_utils import upload_to_google_drive


def process_uploaded_game_film(video_path: str, *, purge_after: bool = False) -> None:
    """Process an uploaded game film video file.

    Parameters
    ----------
    video_path:
        Full path to the video file to process.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(video_path)

    log_dir = Path("output/manual_logs")
    summary_dir = Path("output/summary")
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = None
    try:
        tracker = SmartAutoTracker()
    except Exception:
        tracker = None
    scoreboard = ScoreboardReader()

    playbook_file = (
        "mca_full_playbook_final.json"
        if Path("mca_full_playbook_final.json").exists()
        else "mca_playbook.json"
    )
    playbook = load_playbook(playbook_file)

    frame_logs: List[Dict[str, object]] = []
    jersey_counts: Dict[int, int] = {}
    play_counts: Dict[str, int] = {}
    clip_frames: List = []

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame.shape[1] != 1280 or frame.shape[0] != 720:
            frame = cv2.resize(frame, (1280, 720))

        if tracker is not None:
            x, y, w, h = tracker.track(frame)
        else:
            x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]
        crop = frame[y : y + h, x : x + w]

        jerseys = detect_jerseys(crop)
        for j in jerseys:
            jersey_counts[j] = jersey_counts.get(j, 0) + 1

        state = scoreboard.update(frame)

        frame_logs.append(
            {
                "frame": frame_index,
                "time": frame_index / fps,
                "jerseys": jerseys,
                "box": [int(x), int(y), int(w), int(h)],
                "scoreboard": vars(state),
            }
        )

        clip_frames.append(crop)
        if len(clip_frames) >= 60:
            formation, direction, ptype, mean_flow = detect_play_attributes(clip_frames)
            name, conf = match_play(formation, direction, ptype, mean_flow, playbook)
            play_counts[name] = play_counts.get(name, 0) + 1
            clip_frames.clear()

        if total_frames:
            pct = frame_index / total_frames * 100
            print(f"Processed {frame_index}/{total_frames} frames ({pct:.1f}%)")
        elif frame_index % 100 == 0:
            print(f"Processed {frame_index} frames")

    cap.release()

    log_path = log_dir / f"{path.stem}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(frame_logs, f, indent=2)

    summary = {
        "total_frames": frame_index,
        "total_plays": sum(play_counts.values()),
        "jersey_counts": jersey_counts,
        "play_counts": play_counts,
    }
    summary_path = summary_dir / f"{path.stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    video_ok = upload_to_google_drive(str(path), "GameFilmUploads")
    summary_ok = upload_to_google_drive(str(summary_path), "GameFilmSummaries")

    removed = False
    if purge_after and video_ok and summary_ok:
        try:
            os.remove(path)
            removed = True
        except Exception as exc:  # pragma: no cover - filesystem
            print(f"Failed to delete local video: {exc}")

    print("\nProcessing complete")
    print(f"Total plays detected: {summary['total_plays']}")
    print("Jersey numbers tracked:", sorted(jersey_counts.keys()))
    print("Play types identified:")
    for name, cnt in play_counts.items():
        print(f"  {name}: {cnt}")

    print("\nUpload results:")
    print(f"  Video upload: {'success' if video_ok else 'FAILED'}")
    print(f"  Summary upload: {'success' if summary_ok else 'FAILED'}")
    if purge_after:
        print("  Local video deleted" if removed else "  Local video retained")


__all__ = ["process_uploaded_game_film"]
