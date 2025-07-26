"""Stream camera with smart auto-cropping based on player activity."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

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


def ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        sys.exit("ffmpeg is not installed or not in PATH")
    return path


def select_codec() -> str:
    try:
        output = subprocess.check_output(["ffmpeg", "-encoders"], text=True)
        for codec in ("h264_nvmpi", "h264_nvv4l2enc"):
            if codec in output:
                return codec
    except Exception:
        pass
    return "libx264"


def build_ffmpeg_command(url: str, size: tuple[int, int], fps: float, output: Path) -> list[str]:
    width, height = size
    return [
        ensure_ffmpeg(),
        "-loglevel",
        "verbose",
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
        select_codec(),
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
    ensure_ffmpeg()
    url = os.environ.get("YOUTUBE_RTMP_URL")
    if not url:
        sys.exit("Missing YOUTUBE_RTMP_URL environment variable")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Unable to open camera")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    out_width, out_height = 640, 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0
    print(f"Capture settings: {width}x{height} @ {fps}fps")

    tracker = SmartAutoTracker()

    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("video")
        output_dir.mkdir(exist_ok=True)
        record_file = output_dir / f"game_{timestamp}.mp4"
        cmd = build_ffmpeg_command(url, (out_width, out_height), fps, record_file)
        log_dir = Path("livestream_logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"smart_crop_{timestamp}.log"
        lf = log_file.open("w", encoding="utf-8", errors="ignore")
        print("Running FFmpeg command:", " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=10**8,
        )

        def _reader(pipe, logf):
            for raw in pipe:
                line = raw.decode("utf-8", errors="ignore")
                print(line, end="")
                logf.write(line)

        thread_out = threading.Thread(target=_reader, args=(process.stdout, lf), daemon=True)
        thread_err = threading.Thread(target=_reader, args=(process.stderr, lf), daemon=True)
        thread_out.start()
        thread_err.start()
        first_frame = True
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if first_frame:
                    print("Frame shape:", frame.shape, "dtype:", frame.dtype)
                    cv2.imwrite("debug_frame.jpg", frame)
                    if frame.shape[0] != height or frame.shape[1] != width:
                        print(
                            f"Warning: frame size {frame.shape[1]}x{frame.shape[0]} != {width}x{height}"
                        )
                    if frame.dtype != "uint8" or (len(frame.shape) > 2 and frame.shape[2] != 3):
                        print("Warning: frame is not bgr24 format")
                    first_frame = False
                x, y, w, h = tracker.track(frame)
                crop = frame[y : y + h, x : x + w]
                crop = cv2.resize(crop, (out_width, out_height))
                print(f"Writing frame of shape {crop.shape} to FFmpeg")
                process.stdin.write(crop.astype(np.uint8).tobytes())
        except KeyboardInterrupt:
            pass
        finally:
            if process.stdin:
                process.stdin.close()
            ret = process.wait()
            thread_out.join()
            thread_err.join()
            if process.stderr:
                err_output = process.stderr.read().decode("utf-8", errors="ignore")
                if err_output:
                    print(err_output)
                    lf.write(err_output)
            lf.write(f"\nffmpeg exited with code {ret}\n")
            lf.close()
            if ret != 0:
                print("FFmpeg exited with error:", ret)
        # after each run break; change to continue if restart desired
        break

    cap.release()


if __name__ == "__main__":
    main()
