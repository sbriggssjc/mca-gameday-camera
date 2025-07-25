"""Stream camera with smart auto-cropping based on player activity."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2

from smart_auto_tracker import SmartAutoTracker


def load_env(env_path: str = ".env") -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def build_ffmpeg_command(url: str, size: tuple[int, int], fps: float, output: Path) -> list[str]:
    width, height = size
    return [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        "4500k",
        "-maxrate",
        "4500k",
        "-bufsize",
        "9000k",
        "-g",
        "120",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-f",
        "tee",
        f"[f=flv]{url}|[f=mp4]{output}",
    ]


def main() -> None:
    load_env()
    url = os.environ.get("YOUTUBE_RTMP_URL")
    if not url:
        sys.exit("Missing YOUTUBE_RTMP_URL environment variable")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Unable to open camera")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    tracker = SmartAutoTracker()

    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("video")
        output_dir.mkdir(exist_ok=True)
        record_file = output_dir / f"game_{timestamp}.mp4"
        cmd = build_ffmpeg_command(url, (width, height), fps, record_file)
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                x, y, w, h = tracker.track(frame)
                crop = frame[y : y + h, x : x + w]
                crop = cv2.resize(crop, (width, height))
                process.stdin.write(crop.tobytes())
        except KeyboardInterrupt:
            pass
        finally:
            if process.stdin:
                process.stdin.close()
            process.wait()
        # after each run break; change to continue if restart desired
        break

    cap.release()


if __name__ == "__main__":
    main()
