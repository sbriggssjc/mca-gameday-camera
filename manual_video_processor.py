from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
import zipfile

import cv2

from ai_detector import extract_jersey_number
from play_recognizer import (
    load_playbook,
    detect_play_attributes,
    match_play,
)
from formation_detector import detect_formation
from scoreboard_reader import ScoreboardReader
from smart_auto_tracker import SmartAutoTracker
from gdrive_utils import upload_to_google_drive
from review_queue import add_entry, queue_length


def process_uploaded_game_film(
    video_path: str,
    *,
    purge_after: bool = False,
    max_frames_per_play: int = 2,
    prepare_retrain: bool = False,
) -> None:
    """Process an uploaded game film video file.

    Parameters
    ----------
    video_path:
        Full path to the video file to process.
    prepare_retrain:
        When True, create a training bundle after processing.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(video_path)

    log_dir = Path("output/manual_logs")
    summary_dir = Path("output/summary")
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    training_frame_dir = Path("training/frames")
    training_label_dir = Path("training/labels")
    training_frame_dir.mkdir(parents=True, exist_ok=True)
    training_label_dir.mkdir(parents=True, exist_ok=True)

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
    formation_counts: Dict[str, int] = {}
    player_play_counts: Dict[str, Set[int]] = {}
    clip_frames: List[Tuple[object, int, str, List[int], List[Tuple[int, int, int, int]]]] = []
    play_id = 1
    ocr_failures = 0
    frames_saved = 0

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
        ts = frame_index / fps
        ts_str = f"{int(ts // 3600):02d}:{int((ts % 3600) // 60):02d}:{int(ts % 60):02d}"
        for i, bx in enumerate(boxes):
            num, conf = extract_jersey_number(
                frame,
                bx,
                video_name=path.stem,
                frame_id=frame_index,
                bbox_id=i,
                timestamp=ts_str,
                play_id=play_id,
            )
            if num is not None and conf >= 50.0:
                jerseys.append(int(num))
                jersey_counts[int(num)] = jersey_counts.get(int(num), 0) + 1
                plays = player_play_counts.setdefault(str(num), set())
                plays.add(play_id)
            else:
                ocr_failures += 1

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

        clip_frames.append((crop, frame_index, ts_str, list(jerseys), boxes))
        if len(clip_frames) >= 60:
            frames_only = [cf[0] for cf in clip_frames]
            _, direction, ptype, mean_flow = detect_play_attributes(frames_only)
            formation, _info = detect_formation(clip_frames[0][0], clip_frames[0][4], play_id=play_id, frame_id=clip_frames[0][1])
            name, conf = match_play(formation, direction, ptype, mean_flow, playbook)
            play_counts[name] = play_counts.get(name, 0) + 1
            formation_counts[formation] = formation_counts.get(formation, 0) + 1

            indices: List[int] = []
            if max_frames_per_play > 0:
                indices.append(min(9, len(clip_frames) - 1))
            if max_frames_per_play > 1 and len(clip_frames) > 1:
                indices.append(len(clip_frames) // 2)

            saved_frame = None
            saved_ts = ""
            for idx in indices[:max_frames_per_play]:
                img, fid, fts, jnums, _bxs = clip_frames[idx]
                ts_name = fts.replace(":", "-")
                frame_path = training_frame_dir / f"play_{play_id}_{ts_name}.jpg"
                cv2.imwrite(str(frame_path), img)
                frames_saved += 1
                if saved_frame is None:
                    saved_frame = frame_path
                    saved_ts = fts
                label = {
                    "play_id": play_id,
                    "video": path.name,
                    "timestamp": fts,
                    "play_type": name,
                    "formation": formation,
                    "frame_id": fid,
                    "jersey_numbers": [str(j) for j in jnums],
                    "ball_carrier": "",
                }
                label_path = training_label_dir / f"play_{play_id}_{ts_name}.json"
                with open(label_path, "w", encoding="utf-8") as f:
                    json.dump(label, f, indent=2)

            if name == "unknown" and saved_frame is not None:
                add_entry(
                    {
                        "type": "unknown_play",
                        "frame": str(saved_frame),
                        "play_id": play_id,
                        "timestamp": saved_ts,
                    }
                )

            play_id += 1
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
        "formation_counts": formation_counts,
    }
    participation_counts = {j: len(ids) for j, ids in player_play_counts.items()}
    summary_path = summary_dir / f"{path.stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summaries_dir = Path("output/summaries")
    summaries_dir.mkdir(parents=True, exist_ok=True)
    part_path = summaries_dir / "player_play_counts.json"
    with open(part_path, "w", encoding="utf-8") as f:
        json.dump(participation_counts, f, indent=2)
    print(
        f"\u2705 Player play count summary saved: {part_path}")


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

    learning_dir = Path("training/logs")
    learning_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "video": path.name,
        "plays_detected": summary["total_plays"],
        "jersey_numbers_recognized": len(jersey_counts),
        "jersey_ocr_failures": ocr_failures,
        "unknown_play_types": play_counts.get("unknown", 0),
        "frames_saved": frames_saved,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    stats_path = learning_dir / f"learning_stats_{path.stem}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    global_path = learning_dir / "self_learning_log.json"
    try:
        with open(global_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        if not isinstance(history, list):
            history = []
    except Exception:
        history = []
    history.append(stats)
    with open(global_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\u2705 Learning stats saved: {stats_path}")

    if prepare_retrain:
        bundle_dir = Path("training/bundles")
        bundle_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        bundle_path = bundle_dir / f"retrain_bundle_{ts}.zip"
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder in ["training/frames", "training/labels", "training/uncertain_jerseys"]:
                for root_dir, _, files in os.walk(folder):
                    for file in files:
                        fp = Path(root_dir) / file
                        if fp.is_file():
                            zf.write(fp, fp.relative_to("training"))
        print(f"\u2705 Retraining bundle created: {bundle_path.name}")

    print(
        f"\u26a0\ufe0f  Review queue updated: {queue_length()} items pending in ./training/review_queue.json"
    )


__all__ = ["process_uploaded_game_film"]
