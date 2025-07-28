from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from ai_detector import extract_jersey_number
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

    training_img_dir = Path("training/uncertain_jerseys")
    label_file = Path("training/labels/ocr_review.json")
    training_img_dir.mkdir(parents=True, exist_ok=True)
    label_file.parent.mkdir(parents=True, exist_ok=True)
    ocr_labels: List[Dict[str, object]] = []
    if label_file.exists():
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                ocr_labels = data
        except Exception:
            ocr_labels = []

    playbook_file = (
        "mca_full_playbook_final.json"
        if Path("mca_full_playbook_final.json").exists()
        else "mca_playbook.json"
    )
    playbook = load_playbook(playbook_file)

    frame_logs: List[Dict[str, object]] = []
    jersey_counts: Dict[int, int] = {}
    play_counts: Dict[str, int] = {}
    play_participation: Dict[str, List[int]] = {}
    clip_frames: List = []
    current_play_jerseys: set[str] = set()
    play_id = 1

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
            boxes = [b.as_tuple() for b in tracker.track_boxes.values()]
            if not boxes:
                boxes = [(x, y, x + w, y + h)]
        else:
            x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]
            boxes = [(x, y, x + w, y + h)]
        crop = frame[y : y + h, x : x + w]

        jerseys = []
        for bx in boxes:
            num, conf = extract_jersey_number(frame, bx)
            if num is not None and conf >= 50.0:
                jerseys.append(int(num))
                jersey_counts[int(num)] = jersey_counts.get(int(num), 0) + 1
                current_play_jerseys.add(num)
            else:
                ts = frame_index / fps
                ts_str = f"{int(ts // 3600):02d}_{int((ts % 3600) // 60):02d}_{int(ts % 60):02d}"
                fname = f"{play_id}_{ts_str}.jpg"
                cv2.imwrite(str(training_img_dir / fname), crop)
                cv2.imwrite(str(training_img_dir / f"{play_id}_{ts_str}_full.jpg"), frame)
                ocr_labels.append(
                    {
                        "filename": fname,
                        "bbox": [int(bx[0]), int(bx[1]), int(bx[2] - bx[0]), int(bx[3] - bx[1])],
                        "expected_format": "jersey_number (1–99)",
                        "play_id": play_id,
                        "video_time": ts_str.replace("_", ":"),
                    }
                )

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
            for j in current_play_jerseys:
                plays = play_participation.setdefault(j, [])
                if play_id not in plays:
                    plays.append(play_id)
            play_id += 1
            current_play_jerseys.clear()
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
    participation_counts = {j: len(plays) for j, plays in play_participation.items()}
    summary_path = summary_dir / f"{path.stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summaries_dir = Path("output/summaries")
    summaries_dir.mkdir(parents=True, exist_ok=True)
    part_path = summaries_dir / "player_play_counts.json"
    with open(part_path, "w", encoding="utf-8") as f:
        json.dump(participation_counts, f, indent=2)

    with open(label_file, "w", encoding="utf-8") as f:
        json.dump(ocr_labels, f, indent=2)

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
    print("Player participation counts:")
    for j, cnt in participation_counts.items():
        print(f"  #{j}: {cnt} plays")
    print("Play types identified:")
    for name, cnt in play_counts.items():
        print(f"  {name}: {cnt}")

    print("\nUpload results:")
    print(f"  Video upload: {'success' if video_ok else 'FAILED'}")
    print(f"  Summary upload: {'success' if summary_ok else 'FAILED'}")
    if purge_after:
        print("  Local video deleted" if removed else "  Local video retained")


__all__ = ["process_uploaded_game_film"]
