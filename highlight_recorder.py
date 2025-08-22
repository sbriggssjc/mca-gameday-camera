import cv2
import argparse
import os

# Older OpenCV builds may not expose ``cv2.utils.logging``. Fall back to the
# top-level ``setLogLevel`` function when necessary so that warnings are
# suppressed on both new and old versions.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:  # pragma: no cover - depends on OpenCV version
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
import numpy as np
import time
from datetime import datetime
import subprocess
from pathlib import Path

from scoreboard_reader import ScoreboardReader, ScoreboardState


def open_writer(path: str, fps: float, size: tuple[int, int]):
    """Open H.264 writer, fallback to MJPG if unavailable."""
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        print("avc1 codec not available, falling back to MJPG")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, fps, size)
    return writer


def add_overlay(input_path: str, text: str) -> str:
    """Return path to a new clip with scoreboard overlay."""
    safe_text = text.replace(":", "\\:").replace("'", "\\'")
    input_p = Path(input_path)
    output_path = input_p.with_name(input_p.stem + "_overlay.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_p),
        "-vf",
        (
            "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"\
            f"text='{safe_text}':x=10:y=10:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.5"
        ),
        "-c:a",
        "copy",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"Failed to overlay text with ffmpeg: {exc}")
        return str(input_p)
    return str(output_path)


def monitor(device: str = "/dev/video0", *, output_dir: str = ".", upload: bool = True) -> None:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"Unable to open camera {device}")
        return

    sb_reader = ScoreboardReader()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    ret, prev = cap.read()
    if not ret:
        print("Failed to read initial frame")
        cap.release()
        return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    recording = False
    writer = None
    record_end = 0.0
    clip_path = ""
    overlay_text = ""
    os.makedirs(output_dir, exist_ok=True)
    last_highlight = 0.0
    motion_threshold = 25.0

    print("Monitoring for motion... Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed, stopping.")
                break

            sb_reader.update(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff_val = float(np.mean(cv2.absdiff(prev_gray, gray)))
            now = time.time()

            if diff_val >= motion_threshold:
                if not recording and now - last_highlight >= 5:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    clip_path = os.path.join(output_dir, f"highlight_{timestamp}.mp4")
                    writer = open_writer(clip_path, fps, (1280, 720))
                    if not writer.isOpened():
                        print("Failed to open video writer")
                        break
                    state: ScoreboardState = sb_reader.update(frame)
                    dd = f"{state.down} & {state.distance}" if state.down and state.distance else ""
                    overlay_text = f"Q{state.quarter} {state.clock}"
                    if dd:
                        overlay_text += f" | {dd}"
                    overlay_text += f" | MCA {state.home} - {state.away}"
                    recording = True
                    record_end = now + 10
                    print(f"Motion detected. Recording started: {clip_path}")

            if recording:
                writer.write(frame)
                if now >= record_end:
                    writer.release()
                    recording = False
                    print(f"Clip saved: {clip_path}")
                    clip_with_overlay = add_overlay(clip_path, overlay_text)
                    if upload:
                        result = subprocess.run(
                            ["rclone", "copy", clip_with_overlay, "gdrive:/MCA/GameDayHighlights/"],
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode != 0:
                            print("Upload failed:")
                            print(result.stderr.strip())
                        else:
                            print("Upload successful.")
                    last_highlight = now

            prev_gray = gray
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        if writer and writer.isOpened():
            writer.release()
        print("Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record highlights when motion is detected")
    parser.add_argument("output_dir", nargs="?", default=os.getenv("HIGHLIGHT_DIR", "."), help="directory to store highlight clips")
    parser.add_argument("--device", default="/dev/video0", help="video capture device")
    parser.add_argument("--no-upload", dest="upload", action="store_false", help="disable rclone upload")
    args = parser.parse_args()
    try:
        monitor(device=args.device, output_dir=args.output_dir, upload=args.upload)
    except Exception as exc:
        print(f"Error: {exc}")

