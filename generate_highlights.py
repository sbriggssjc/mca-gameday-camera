import argparse
import json
import os
from collections import deque

import cv2

from overlay_engine import OverlayEngine
from scoreboard_reader import ScoreboardReader, ScoreboardState


def open_writer(path: str, fps: float, size: tuple[int, int]):
    """Open H.264 writer, fallback to MJPG if necessary."""
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, fps, size)
    return writer


def parse_clock(clock: str) -> int:
    """Convert MM:SS string to seconds."""
    try:
        m, s = clock.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return 0


def generate(video_path: str, output_dir: str = "highlights") -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    os.makedirs(output_dir, exist_ok=True)

    overlay = OverlayEngine()
    reader = ScoreboardReader()

    buffer = deque(maxlen=int(fps * 5))
    writer = None
    end_time = 0.0
    clip_id = 1
    last_state: ScoreboardState | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        state = reader.update(frame)
        overlay.draw(frame, state)
        buffer.append(frame.copy())

        if last_state and (state.home != last_state.home or state.away != last_state.away):
            clip_path = os.path.join(output_dir, f"highlight_{clip_id}.mp4")
            writer = open_writer(clip_path, fps, (width, height))
            for bf in buffer:
                writer.write(bf)
            end_time = t + 5
            meta = {
                "play_id": clip_id,
                "score": f"{state.home}-{state.away}",
                "time": state.clock,
                "down": state.down,
            }
            meta_path = os.path.join(output_dir, f"highlight_{clip_id}.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            clip_id += 1

        if writer:
            writer.write(frame)
            if t >= end_time:
                writer.release()
                writer = None
                buffer.clear()

        last_state = state

    if writer:
        writer.release()
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate highlight clips with overlays")
    parser.add_argument("video", help="Raw game video file")
    parser.add_argument("--output", default="highlights", help="Directory for highlight clips")
    args = parser.parse_args()
    generate(args.video, args.output)


if __name__ == "__main__":
    main()
