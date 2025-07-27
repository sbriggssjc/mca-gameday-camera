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

# Stream resolution constants
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720

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


def build_ffmpeg_command(
    url: str,
    size: tuple[int, int],
    fps: float,
    output: Path,
    *,
    filters: str | None = None,
    bitrate: str,
    maxrate: str,
    bufsize: str,

) -> list[str]:
    width, height = size
    cmd = [
        ensure_ffmpeg(),
        "-loglevel",
        "verbose",
        "-y",
        "-thread_queue_size",
        "512",
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
        "-thread_queue_size",
        "512",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
    ]
    if filters:
        cmd.extend(["-vf", filters])
    cmd += [
        "-b:v",
        bitrate,
        "-maxrate",
        maxrate,
        "-bufsize",
        bufsize,
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
    return cmd


def build_v4l2_command(
    url: str,
    size: tuple[int, int],
    fps: float,
    output: Path,
    *,
    device: str = "/dev/video0",
    filters: str | None = None,
    bitrate: str,
    maxrate: str,
    bufsize: str,

) -> list[str]:
    width, height = size
    cmd = [
        ensure_ffmpeg(),
        "-loglevel",
        "verbose",
        "-y",
        "-thread_queue_size",
        "512",
        "-f",
        "v4l2",
        "-framerate",
        str(int(fps)),
        "-video_size",
        f"{width}x{height}",
        "-i",
        device,
        "-thread_queue_size",
        "512",
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
    ]
    if filters:
        cmd.extend(["-vf", filters])
    cmd += [
        "-b:v",
        bitrate,
        "-maxrate",
        maxrate,
        "-bufsize",
        bufsize,
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
    return cmd


def build_record_command(
    size: tuple[int, int],
    fps: float,
    output: Path,
    *,
    device: str = "/dev/video0",
    filters: str | None = None,
    bitrate: str,
    maxrate: str,
    bufsize: str,
) -> list[str]:
    """Record from the camera directly to a local MP4 file."""
    width, height = size
    cmd = [
        ensure_ffmpeg(),
        "-loglevel",
        "verbose",
        "-y",
        "-thread_queue_size",
        "512",
        "-f",
        "v4l2",
        "-framerate",
        str(int(fps)),
        "-video_size",
        f"{width}x{height}",
        "-i",
        device,
        "-thread_queue_size",
        "512",
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
    ]
    if filters:
        cmd.extend(["-vf", filters])
    cmd += [
        "-b:v",
        bitrate,
        "-maxrate",
        maxrate,
        "-bufsize",
        bufsize,
        "-g",
        "120",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output),
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-direct", action="store_true", help="run direct FFmpeg test without OpenCV")
    parser.add_argument("--debug", action="store_true", help="save a few frames before streaming")
    parser.add_argument(
        "--output-size",
        default=os.environ.get("OUTPUT_RESOLUTION"),
        help="stream resolution WxH, defaults to camera size or OUTPUT_RESOLUTION env",
    )
    parser.add_argument(
        "--resolution",
        default=f"{STREAM_WIDTH}x{STREAM_HEIGHT}",
        help="stream resolution WxH, e.g. 1280x720 or 1920x1080",
    )
    parser.add_argument(
        "--bitrate",
        default="13500k",
        help="target video bitrate",
    )
    parser.add_argument(
        "--maxrate",
        default="13500k",
        help="maximum video bitrate",
    )
    parser.add_argument(
        "--bufsize",
        default="27000k",
        help="encoder buffer size",
    )
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or STREAM_WIDTH)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or STREAM_HEIGHT)
    out_width, out_height = STREAM_WIDTH, STREAM_HEIGHT
    filter_str = (
        f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={STREAM_WIDTH}:{STREAM_HEIGHT}"
    )
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    state_reader = ScoreboardReader()
    overlay = OverlayEngine()

    # Calibrate scoreboard ROI once at startup
    ret, calib_frame = cap.read()
    if not ret or calib_frame is None:
        sys.exit("Unable to read from camera")
    if calib_frame.shape[0] != STREAM_HEIGHT or calib_frame.shape[1] != STREAM_WIDTH:
        raise RuntimeError(
            f"Camera returned unexpected resolution {calib_frame.shape[1]}x{calib_frame.shape[0]}"
        )
    if np.std(calib_frame) < 2 or np.allclose(calib_frame[:, :, 1], 255, atol=5):
        raise RuntimeError("Camera frame appears invalid or all green")
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
            filters=filter_str,
            bitrate=args.bitrate,
            maxrate=args.maxrate,
            bufsize=args.bufsize,
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, out_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, out_height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or out_width)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or out_height)
    if width != out_width or height != out_height:
        print(
            f"Warning: requested resolution {out_width}x{out_height} not supported, using {width}x{height}"
        )
    print(f"Capture settings: {width}x{height} @ {fps}fps")
    print(f"Output resolution: {out_width}x{out_height}")
    tracker = None
    try:
        tracker = AutoTracker()
    except Exception:
        tracker = None

    log_dir = Path("livestream_logs")
    log_dir.mkdir(exist_ok=True)

    output_dir = Path("video")
    output_dir.mkdir(exist_ok=True)

    retries = 0
    max_retries = 1
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"stream_{timestamp}.log"
        record_file = output_dir / f"game_{timestamp}.mp4"
        with log_file.open("w", encoding="utf-8", errors="ignore") as lf:

            lf.write(f"Input resolution: {width}x{height}\n")
            lf.write(f"Output resolution: {out_width}x{out_height}\n")
            print(f"Input resolution: {width}x{height}")
            print(f"Output resolution: {out_width}x{out_height}")

            cmd = build_ffmpeg_command(
                url,
                (out_width, out_height),
                fps,
                record_file,

                filters=filter_str,

                bitrate=args.bitrate,
                maxrate=args.maxrate,
                bufsize=args.bufsize,

            )
            pix_fmt = "bgr24"
            if "-pix_fmt" in cmd:
                try:
                    pix_fmt = cmd[cmd.index("-pix_fmt") + 1]
                except Exception:
                    pass
            cmd_str = " ".join(cmd)
            print("Running FFmpeg command:", cmd_str)
            lf.write("Running FFmpeg command: " + cmd_str + "\n")
            lf.write(f"Audio enabled: {'-c:a' in cmd}\n")
            lf.write(f"Video enabled: {'-c:v' in cmd}\n")
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
                    if "Output file does not contain any stream" in line:
                        logf.write("ALERT: no stream detected\n")
                    if "Broken pipe" in line:
                        logf.write("ALERT: broken pipe\n")
                    if "drop=" in line:
                        logf.write("ALERT: dropped frames reported\n")
                    match = re.search(r"delay\s*=\s*(\d+)\w*", line)
                    if match and int(match.group(1)) > 500:
                        logf.write("ALERT: frame delay > 500ms\n")

            thread_out = threading.Thread(target=_reader, args=(process.stdout, lf), daemon=True)
            thread_err = threading.Thread(target=_reader, args=(process.stderr, lf), daemon=True)
            thread_out.start()
            thread_err.start()
            first_frame = True
            frame_count = 0
            frame_interval = 1.0 / fps
            next_frame_time = time.time()
            try:
                while True:
                    now = time.time()
                    if now - next_frame_time > frame_interval * 1.5:
                        warn = f"Frame delay {now - next_frame_time:.3f}s"
                        print(warn)
                        lf.write(warn + "\n")
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        lf.write("Dropped frame\n")
                        print("Dropped frame")
                        next_frame_time += frame_interval
                        continue
                    next_frame_time += frame_interval
                    if frame.shape[0] != height or frame.shape[1] != width:
                        warn = (
                            f"Unexpected frame size {frame.shape[1]}x{frame.shape[0]}"
                        )
                        print(warn)
                        lf.write(warn + "\n")
                    if frame.shape[2] == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
                    elif frame.shape[2] == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    if frame.shape[2] != 3 or frame.dtype != np.uint8:
                        raise ValueError(
                            f"Unexpected frame shape: {frame.shape} or dtype: {frame.dtype}"
                        )
                    if first_frame:
                        if np.std(frame) < 2:
                            msg = "Skipping blank first frame"
                            print(msg)
                            lf.write(msg + "\n")
                            continue
                        print("Frame shape:", frame.shape, "dtype:", frame.dtype)
                        if args.debug:
                            cv2.imwrite("frame_debug_0.jpg", frame)
                        first_frame = False
                    print(
                        f"Frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}"
                    )
                    if tracker:
                        x, y, w, h = tracker.track(frame)
                        frame = frame[y : y + h, x : x + w]
                        frame = cv2.resize(frame, (width, height))
                    state = state_reader.update(frame)
                    overlay.draw(frame, state)
                    cv2.imshow("Debug Preview", frame)
                    cv2.waitKey(1)
                    frame_resized = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
                    if args.debug and frame_count < 5:
                        cv2.imwrite(f"frame_debug_{frame_count+1}.jpg", frame_resized)
                    if frame_resized.shape != (STREAM_HEIGHT, STREAM_WIDTH, 3) or frame_resized.dtype != np.uint8:
                        raise ValueError(
                            f"Resized frame has shape {frame_resized.shape} and dtype {frame_resized.dtype}"
                        )
                    print(f"Writing frame of shape {frame_resized.shape} to FFmpeg")
                    try:
                        if pix_fmt == "rgb24":
                            frame_conv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                            process.stdin.write(frame_conv.astype(np.uint8).tobytes())
                        else:
                            process.stdin.write(frame_resized.astype(np.uint8).tobytes())
                        frame_count += 1
                        if frame_count % 30 == 0:
                            print(f"Sent {frame_count} frames to FFmpeg")
                    except BrokenPipeError:
                        msg = "FFmpeg closed the pipe unexpectedly. Aborting stream."
                        print(msg)
                        lf.write(msg + "\n")
                        break
                    sleep_time = next_frame_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            except KeyboardInterrupt:
                pass
            finally:
                if process.stdin:
                    process.stdin.close()
                ret = process.wait()
                thread_out.join()
                thread_err.join()
                cv2.destroyAllWindows()
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
                    test_cmd = build_v4l2_command(
                        url,
                        (width, height),
                        fps,
                        record_file,
                        device=device,

                        filters=filter_str,

                        bitrate=args.bitrate,
                        maxrate=args.maxrate,
                        bufsize=args.bufsize,

                    )
                    lf.write("Running: " + " ".join(test_cmd) + "\n")
                    result = subprocess.run(test_cmd, capture_output=True, text=True)
                    lf.write(result.stdout)
                    lf.write(result.stderr)
                    print("FFmpeg output:")
                    print(result.stdout)
                    print(result.stderr)

                    lf.write("\nTesting camera by recording locally...\n")
                    file_cmd = build_record_command(
                        (STREAM_WIDTH, STREAM_HEIGHT),
                        fps,
                        Path("output.mp4"),
                        device=device,

                        filters=filter_str,

                        bitrate=args.bitrate,
                        maxrate=args.maxrate,
                        bufsize=args.bufsize,

                    )
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
        elif retries < max_retries:
            retries += 1
            print(f"FFmpeg failed with code {ret}. Restarting (attempt {retries})")
            time.sleep(2)
            continue
        time.sleep(5)
        
    cap.release()


if __name__ == "__main__":
    main()
