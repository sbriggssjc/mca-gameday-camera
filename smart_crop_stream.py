"""Stream camera with smart auto-cropping based on player activity."""

from __future__ import annotations

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from ffmpeg_utils import build_ffmpeg_args, detect_encoder
from queue import Queue, Full


import argparse
import cv2
import numpy as np
import time

from smart_auto_tracker import SmartAutoTracker
from upload_to_drive import upload_to_drive


def upload_after_stream(video_path: Path, folder_id: str) -> None:
    """Upload ``video_path`` to Google Drive once streaming finishes."""
    logging.info("Streaming finished, uploading to Drive...")
    try:
        file_id, url_view = upload_to_drive(video_path, folder_id)
        uploaded_dir = Path("video/uploaded")
        uploaded_dir.mkdir(exist_ok=True)
        video_path.rename(uploaded_dir / video_path.name)
        logging.info(f"Upload successful: {video_path.name} -> {file_id}")
    except ImportError:
        logging.info(
            "PyDrive is not installed. Run `pip install PyDrive google-api-python-client oauth2client` to enable Google Drive uploads."
        )
    except Exception as exc:
        logging.info(f"Upload failed: {exc}")
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


def log_ffmpeg_stderr(stderr, log_file=None, buffer=None) -> None:
    """Continuously read and print FFmpeg stderr."""
    for line in stderr:
        text = line.decode("utf-8", errors="ignore")
        if buffer is not None:
            buffer.append(text)
        if log_file is not None:
            log_file.write(text)
            log_file.flush()
        logging.info("[FFMPEG]", text, end="")
def build_ffmpeg_command(
    url: str,
    size: tuple[int, int],
    fps: float,
    output: Path,
    video_encoder: str,
) -> list[str]:
    width, height = size
    return build_ffmpeg_args(
        video_source="-",
        audio_device=None,
        output_url=f"[f=flv]{url}|[f=mp4]{output}",
        audio_gain_db=0.0,
        resolution=f"{width}x{height}",
        framerate=int(fps),
        video_codec=video_encoder,
        video_is_pipe=True,
        extra_args=[
            "-maxrate",
            "4500k",
            "-bufsize",
            "9000k",
            "-g",
            "120",
            "-f",
            "tee",
        ],
    )


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
    logging.info(f"Capture settings: {width}x{height} @ {fps}fps")
    tracker = SmartAutoTracker()

    try:
        video_encoder = detect_encoder()
    except RuntimeError as exc:
        logging.info(exc)
        return
    logging.info("[INFO] Streaming raw BGR → FFmpeg rawvideo → RTMP using", video_encoder)
    retries = 0
    max_retries = 1
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("video")
        output_dir.mkdir(exist_ok=True)
        record_file = output_dir / f"game_{timestamp}.mp4"
        cmd = build_ffmpeg_command(url, (out_width, out_height), fps, record_file, video_encoder)
        log_dir = Path("livestream_logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"smart_crop_{timestamp}.log"
        lf = log_file.open("w", encoding="utf-8", errors="ignore")
        cmd_str = " ".join(cmd)
        logging.info("Starting FFmpeg stream...")
        logging.info("Running FFmpeg command:", cmd_str)
        lf.write("Running FFmpeg command: " + cmd_str + "\n")
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=10**8,
        )

        stderr_lines: list[str] = []

        def _reader(pipe, logf):
            for raw in pipe:
                line = raw.decode("utf-8", errors="ignore")
                logging.info(line, end="")
                logf.write(line)

        thread_out = threading.Thread(target=_reader, args=(process.stdout, lf), daemon=True)
        thread_err = threading.Thread(
            target=log_ffmpeg_stderr, args=(process.stderr, lf, stderr_lines), daemon=True
        )
        thread_out.start()
        thread_err.start()

        start = time.time()
        while time.time() - start < 15:
            if process.poll() is not None:
                err = "".join(stderr_lines)
                logging.info("FFmpeg failed to launch. Exiting...", err)
                lf.write(err)
                lf.close()
                return
            time.sleep(0.5)

        frame_queue: Queue[bytes] = Queue(maxsize=30)

        def encode_worker() -> None:
            frame_interval = 1.0 / fps
            frame_count = 0
            while True:
                start_time = time.time()
                frame_bytes = frame_queue.get()
                if frame_bytes is None:
                    break
                try:
                    process.stdin.write(frame_bytes)
                    process.stdin.flush()
                    frame_count += 1
                    if frame_count % 100 == 0:
                        logging.info(f"🟢 {frame_count} frames sent to FFmpeg")
                except BrokenPipeError:
                    logging.info("❌ FFmpeg pipe broken — exiting stream loop")
                    break
                elapsed = time.time() - start_time
                delay = frame_interval - elapsed
                if delay > 0:
                    time.sleep(delay)

        encoder_thread = threading.Thread(target=encode_worker, daemon=True)
        encoder_thread.start()

        first_frame = True
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if first_frame:
                    logging.info("Frame shape:", frame.shape, "dtype:", frame.dtype)
                    cv2.imwrite("debug_frame.jpg", frame)
                    if frame.shape[0] != height or frame.shape[1] != width:
                        logging.info(
                            f"Warning: frame size {frame.shape[1]}x{frame.shape[0]} != {width}x{height}"
                        )
                    if frame.dtype != "uint8" or (len(frame.shape) > 2 and frame.shape[2] != 3):
                        logging.info("Warning: frame is not bgr24 format")
                    first_frame = False
                x, y, w, h = tracker.track(frame)
                crop = frame[y : y + h, x : x + w]
                crop = cv2.resize(crop, (out_width, out_height))
                if crop.shape != (out_height, out_width, 3) or crop.dtype != np.uint8:
                    raise ValueError(f"Crop frame has shape {crop.shape} and dtype {crop.dtype}")
                logging.info(f"Queuing frame of shape {crop.shape} for FFmpeg")
                try:
                    frame_yuv = cv2.cvtColor(crop, cv2.COLOR_BGR2YUV_I420)
                    frame_queue.put_nowait(frame_yuv.tobytes())
                except Full:
                    logging.info("[WARNING] Encoding queue full; dropping frame")
        except KeyboardInterrupt:
            pass
        finally:
            frame_queue.put(None)
            encoder_thread.join()
            if process.stdin:
                process.stdin.close()
            ret = process.wait()
            thread_out.join()
            thread_err.join()
            err_output = "".join(stderr_lines)
            if err_output:
                logging.info(err_output)
                lf.write(err_output)
            lf.write(f"\nffmpeg exited with code {ret}\n")
            lf.close()
            if ret != 0:
                logging.info("FFmpeg exited with error:", ret)
        if ret != 0 and retries < max_retries:
            retries += 1
            logging.info(f"FFmpeg failed with code {ret}. Restarting (attempt {retries})")
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
