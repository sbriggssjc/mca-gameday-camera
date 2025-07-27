"""Stream camera with smart auto-cropping based on player activity."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import argparse
import cv2
import numpy as np
import time

from smart_auto_tracker import SmartAutoTracker
from upload_to_drive import upload_to_drive


def upload_after_stream(video_path: Path, folder_id: str) -> None:
    """Upload ``video_path`` to Google Drive once streaming finishes."""
    print("Streaming finished, uploading to Drive...")
    try:
        file_id, url_view = upload_to_drive(video_path, folder_id)
        uploaded_dir = Path("video/uploaded")
        uploaded_dir.mkdir(exist_ok=True)
        video_path.rename(uploaded_dir / video_path.name)
        print(f"Upload successful: {video_path.name} -> {file_id}")
    except ImportError:
        print(
            "PyDrive is not installed. Run `pip install PyDrive google-api-python-client oauth2client` to enable Google Drive uploads."
        )
    except Exception as exc:
        print(f"Upload failed: {exc}")


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
        if "h264_nvmpi" in output:
            return "h264_nvmpi"
    except Exception:
        pass
    return "libx264"


def log_ffmpeg_stderr(stderr, log_file=None) -> None:
    """Continuously read and print FFmpeg stderr."""
    for line in stderr:
        text = line.decode("utf-8", errors="ignore")
        if log_file is not None:
            log_file.write(text)
        print("[FFMPEG]", text, end="")


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
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip Google Drive upload of the recording",
    )
    parser.add_argument(
        "--folder-id",
        default=os.getenv("GDRIVE_FOLDER_ID"),
        help="Destination Google Drive folder ID",
    )
    args = parser.parse_args()

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

    retries = 0
    max_retries = 1
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
        cmd_str = " ".join(cmd)
        print("Starting FFmpeg stream...")
        print("Running FFmpeg command:", cmd_str)
        lf.write("Running FFmpeg command: " + cmd_str + "\n")
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=10**8,
        )
        if process.poll() is not None:
            print("FFmpeg failed to launch. Exiting...")
            return

        def _reader(pipe, logf):
            for raw in pipe:
                line = raw.decode("utf-8", errors="ignore")
                print(line, end="")
                logf.write(line)

        thread_out = threading.Thread(target=_reader, args=(process.stdout, lf), daemon=True)
        thread_err = threading.Thread(target=log_ffmpeg_stderr, args=(process.stderr, lf), daemon=True)
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
                if crop.shape != (out_height, out_width, 3) or crop.dtype != np.uint8:
                    raise ValueError(f"Crop frame has shape {crop.shape} and dtype {crop.dtype}")
                print(f"Writing frame of shape {crop.shape} to FFmpeg")
                process.stdin.write(crop.astype(np.uint8).tobytes())
        except BrokenPipeError:
            print("FFmpeg pipe closed unexpectedly.")
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
        if ret != 0 and retries < max_retries:
            retries += 1
            print(f"FFmpeg failed with code {ret}. Restarting (attempt {retries})")
            time.sleep(2)
            continue
        # after each run break
        break

    cap.release()

    if ret == 0 and not args.no_upload:
        folder = args.folder_id or os.getenv("GDRIVE_FOLDER_ID")
        if folder is not None:
            upload_thread = threading.Thread(
                target=upload_after_stream, args=(record_file, folder)
            )
            upload_thread.start()
            upload_thread.join()


if __name__ == "__main__":
    main()
