import os
import re
import shutil
import subprocess
import sys
import time
import threading
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from auto_tracker import AutoTracker
from overlay_engine import OverlayEngine
from scoreboard_reader import ScoreboardReader
from game_uploader import upload_game


def check_device_free(device: str = "/dev/video0") -> None:
    """Exit with instructions if the device is still in use."""
    cmds = [["lsof", device], ["fuser", device]]
    for cmd in cmds:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            continue
        output = (result.stdout or result.stderr).strip()
        if output:
            print(f"Processes using {device}:")
            print(output)
            print(f"{device} is busy. Try 'sudo fuser -k {device}' or reboot.")
            sys.exit(1)
        break


def wait_for_device(device: str = "/dev/video0") -> None:
    """Block until the video device is free."""
    while os.system(f"lsof {device} > /dev/null 2>&1") == 0:
        time.sleep(1)


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


def build_v4l2_command(
    url: str,
    size: tuple[int, int],
    fps: float,
    output: Path,
    device: str = "/dev/video0",
) -> list[str]:
    width, height = size
    return [
        ensure_ffmpeg(),
        "-loglevel",
        "verbose",
        "-y",
        "-f",
        "v4l2",
        "-framerate",
        str(int(fps)),
        "-video_size",
        f"{width}x{height}",
        "-i",
        device,
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


def build_record_command(
    size: tuple[int, int],
    fps: float,
    output: Path,
    device: str = "/dev/video0",
) -> list[str]:
    """Record from the camera directly to a local MP4 file."""
    width, height = size
    return [
        ensure_ffmpeg(),
        "-loglevel",
        "verbose",
        "-y",
        "-f",
        "v4l2",
        "-framerate",
        str(int(fps)),
        "-video_size",
        f"{width}x{height}",
        "-i",
        device,
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
        str(output),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-direct", action="store_true", help="run direct FFmpeg test without OpenCV")
    args = parser.parse_args()

    load_env()
    ensure_ffmpeg()
    url = os.environ.get("YOUTUBE_RTMP_URL")
    if not url:
        sys.exit("Missing YOUTUBE_RTMP_URL environment variable")
    validate_rtmp_url(url)

    # Check network connectivity
    if subprocess.call(["ping", "-c", "1", "youtube.com"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        sys.exit("Unable to reach youtube.com. Check network connection.")

    device = os.environ.get("VIDEO_DEVICE", "/dev/video0")
    wait_for_device(device)
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        sys.exit(f"Unable to open camera {device}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    out_width, out_height = 640, 480
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

    # Release the camera after ROI selection so FFmpeg can access it
    cap.release()
    cv2.destroyAllWindows()
    time.sleep(2)
    wait_for_device(device)

    if args.test_direct:
        test_cmd = build_v4l2_command(
            url,
            (out_width, out_height),
            fps,
            Path("output_test.mp4"),
            device=device,
        )
        print("Running direct FFmpeg test command:", " ".join(test_cmd))
        subprocess.run(test_cmd)
        return

    # Safety check to ensure the device is free before starting FFmpeg
    check_device_free(device)

    check_cap = cv2.VideoCapture(device)
    if not check_cap.isOpened():
        sys.exit("Camera is busy.")
    check_cap.release()

    wait_for_device(device)

    # Re-open the camera for actual streaming
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        sys.exit(f"Unable to reopen camera {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    print(f"Capture settings: {width}x{height} @ {fps}fps")
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
        with log_file.open("w", encoding="utf-8", errors="ignore") as lf:
            cmd = build_ffmpeg_command(url, (out_width, out_height), fps, record_file)
            pix_fmt = "bgr24"
            if "-pix_fmt" in cmd:
                try:
                    pix_fmt = cmd[cmd.index("-pix_fmt") + 1]
                except Exception:
                    pass
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
                            print(f"Warning: frame is not {pix_fmt} format")
                        first_frame = False
                    if tracker:
                        x, y, w, h = tracker.track(frame)
                        frame = frame[y : y + h, x : x + w]
                        frame = cv2.resize(frame, (width, height))
                    state = state_reader.update(frame)
                    overlay.draw(frame, state)
                    frame_resized = cv2.resize(frame, (out_width, out_height))
                    print(f"Writing frame of shape {frame_resized.shape} to FFmpeg")
                    try:
                        if pix_fmt == "rgb24":
                            frame_conv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                            process.stdin.write(frame_conv.astype(np.uint8).tobytes())
                        else:
                            process.stdin.write(frame_resized.astype(np.uint8).tobytes())
                    except BrokenPipeError:
                        msg = "FFmpeg closed the pipe unexpectedly. Aborting stream."
                        print(msg)
                        lf.write(msg + "\n")
                        break
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
                if ret != 0:
                    print("FFmpeg exited with error:", ret)
                    lf.write("\nFFmpeg failed. Running diagnostics...\n")
                    cap.release()
                    cv2.destroyAllWindows()
                    test_cmd = build_v4l2_command(url, (width, height), fps, record_file, device=device)
                    lf.write("Running: " + " ".join(test_cmd) + "\n")
                    result = subprocess.run(test_cmd, capture_output=True, text=True)
                    lf.write(result.stdout)
                    lf.write(result.stderr)
                    print("FFmpeg output:")
                    print(result.stdout)
                    print(result.stderr)

                    lf.write("\nTesting camera by recording locally...\n")
                    file_cmd = build_record_command((out_width, out_height), fps, Path("output.mp4"), device=device)
                    lf.write("Running: " + " ".join(map(str, file_cmd)) + "\n")
                    record_result = subprocess.run(file_cmd, capture_output=True, text=True)
                    lf.write(record_result.stdout)
                    lf.write(record_result.stderr)
                    print(record_result.stdout)
                    print(record_result.stderr)
                    break
        if ret == 0:
            try:
                upload_game(str(record_file))
            except Exception as exc:
                with log_file.open("a", encoding="utf-8", errors="ignore") as lf:
                    lf.write(f"\nUpload failed: {exc}\n")
            break
        time.sleep(5)

    cap.release()


if __name__ == "__main__":
    main()
