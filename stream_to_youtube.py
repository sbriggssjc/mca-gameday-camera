import os
import re
import shutil
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2

from auto_tracker import AutoTracker
from overlay_engine import OverlayEngine
from scoreboard_reader import ScoreboardReader
from game_uploader import upload_game


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
    """Return path to ffmpeg or exit if not installed."""
    path = shutil.which("ffmpeg")
    if not path:
        sys.exit("ffmpeg is not installed or not in PATH")
    return path


def validate_rtmp_url(url: str) -> None:
    """Exit if the RTMP URL does not match the expected YouTube pattern."""
    pattern = r"^rtmp://a\.rtmp\.youtube\.com/live2/[A-Za-z0-9_-]+$"
    if not re.match(pattern, url):
        sys.exit(
            "Invalid YOUTUBE_RTMP_URL. It should look like "
            "rtmp://a.rtmp.youtube.com/live2/YOUR_STREAM_KEY"
        )


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
        f"[f=flv]{url}|[f=mp4]{output}"
    ]


def main() -> None:
    load_env()
    ensure_ffmpeg()
    url = os.environ.get("YOUTUBE_RTMP_URL")
    if not url:
        sys.exit("Missing YOUTUBE_RTMP_URL environment variable")
    validate_rtmp_url(url)

    # Check network connectivity
    if subprocess.call(["ping", "-c", "1", "youtube.com"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        sys.exit("Unable to reach youtube.com. Check network connection.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Unable to open camera")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    state_reader = ScoreboardReader()
    overlay = OverlayEngine()

    # Calibrate scoreboard ROI once at startup
    ret, calib_frame = cap.read()
    if not ret:
        sys.exit("Unable to read from camera")
    if state_reader.roi is None:
        state_reader.calibrate(calib_frame)
    tracker = None
    try:
        tracker = AutoTracker()
    except Exception:
        tracker = None

    log_dir = Path("livestream_logs")
    log_dir.mkdir(exist_ok=True)

    output_dir = Path("video")
    output_dir.mkdir(exist_ok=True)

    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"stream_{timestamp}.log"
        record_file = output_dir / f"game_{timestamp}.mp4"
        with log_file.open("w") as lf:
            cmd = build_ffmpeg_command(url, (width, height), fps, record_file)
            print("Running FFmpeg command:", " ".join(cmd))
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            def _reader(pipe, logf):
                for line in pipe:
                    print(line, end="")
                    logf.write(line)

            thread = threading.Thread(target=_reader, args=(process.stdout, lf), daemon=True)
            thread.start()
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if tracker:
                        x, y, w, h = tracker.track(frame)
                        frame = frame[y : y + h, x : x + w]
                        frame = cv2.resize(frame, (width, height))
                    state = state_reader.update(frame)
                    overlay.draw(frame, state)
                    process.stdin.write(frame.tobytes())
            except KeyboardInterrupt:
                pass
            finally:
                if process.stdin:
                    process.stdin.close()
                ret = process.wait()
                thread.join()
                lf.write(f"\nffmpeg exited with code {ret}\n")
                if ret != 0:
                    print("FFmpeg exited with error:", ret)
        if ret == 0:
            try:
                upload_game(str(record_file))
            except Exception as exc:
                with log_file.open("a") as lf:
                    lf.write(f"\nUpload failed: {exc}\n")
            break
        time.sleep(5)

    cap.release()


if __name__ == "__main__":
    main()
