import os
import re
import shutil
import subprocess
import sys
import time
import threading
import argparse
import select
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Optional utility to query camera resolution via v4l2-ctl
def get_camera_resolution(device: str = "/dev/video0") -> tuple[int, int]:
    """Return (width, height) using v4l2-ctl if available."""
    v4l2ctl = shutil.which("v4l2-ctl")
    if not v4l2ctl:
        return STREAM_WIDTH, STREAM_HEIGHT
    try:
        output = subprocess.check_output([
            v4l2ctl,
            "-d",
            device,
            "--get-fmt-video",
        ], text=True)
        match = re.search(r"Width/Height\s*:\s*(\d+)/(\d+)", output)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    return STREAM_WIDTH, STREAM_HEIGHT

# Stream resolution constants
STREAM_WIDTH = 1920
STREAM_HEIGHT = 1080
STREAM_FPS = 30.0

from auto_tracker import AutoTracker
from overlay_engine import OverlayEngine
from scoreboard_reader import ScoreboardReader
from upload_to_drive import upload_to_drive


def upload_after_stream(video_path: Path, folder_id: str) -> None:
    """Upload ``video_path`` to Google Drive once streaming finishes."""
    print("Streaming finished, uploading to Drive...")
    try:
        file_id, url_view = upload_to_drive(video_path, folder_id)
        uploaded_dir = Path("video/uploaded")
        uploaded_dir.mkdir(exist_ok=True)
        video_path.rename(uploaded_dir / video_path.name)
        print(f"Upload successful: {url_view}")
    except ImportError:
        print(
            "PyDrive is not installed. Run `pip install PyDrive google-api-python-client oauth2client` to enable Google Drive uploads."
        )
    except Exception as exc:
        print(f"Upload failed: {exc}")


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


def select_codec() -> str:
    """Return the software H.264 encoder."""
    return "libx264"


def log_ffmpeg_stderr(stderr, log_file=None, buffer=None) -> None:
    """Continuously read and print FFmpeg stderr.

    Parameters
    ----------
    stderr : IO[Any]
        The stderr pipe from FFmpeg.
    log_file : IO[str] | None
        Optional log file to write stderr output to.
    buffer : collections.deque[str] | None
        Optional deque used to store recent stderr lines for later
        inspection if FFmpeg exits early.
    """
    for line in stderr:
        text = line.decode("utf-8", errors="ignore")
        if buffer is not None:
            buffer.append(text)
        if log_file is not None:
            log_file.write(text)
        print("[FFMPEG]", text, end="")


def suggest_ffmpeg_exit_causes() -> None:
    """Print common reasons FFmpeg might exit immediately."""
    print("Possible reasons for early FFmpeg termination:")
    print("- Invalid input format (e.g., incorrect OpenCV frame encoding)")
    print("- Bad or missing input resolution or frame rate")
    print("- Stream key or RTMP URL problems")
    print("- Missing ffmpeg binary or codecs")


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
    codec = select_codec()
    cmd = [
        ensure_ffmpeg(),
        "-re",
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
        "-framerate",
        str(int(fps)),
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-thread_queue_size",
        "512",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=stereo",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        codec,
    ]
    if codec == "libx264":
        cmd += [
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-x264-params",
            "bframes=0",
        ]
    cmd += ["-pix_fmt", "yuv420p", "-r", str(int(fps))]
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
        "30",
        "-keyint_min",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-sample_fmt",
        "fltp",
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
    codec = select_codec()
    cmd = [
        ensure_ffmpeg(),
        "-re",
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
        "-r",
        str(int(fps)),
        "-i",
        device,
        "-thread_queue_size",
        "512",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=stereo",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        codec,
    ]
    if codec == "libx264":
        cmd += [
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-x264-params",
            "bframes=0",
        ]
    cmd += ["-pix_fmt", "yuv420p", "-r", str(int(fps))]
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
        "30",
        "-keyint_min",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-sample_fmt",
        "fltp",
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
    codec = select_codec()
    cmd = [
        ensure_ffmpeg(),
        "-re",
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
        "-r",
        str(int(fps)),
        "-i",
        device,
        "-thread_queue_size",
        "512",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=stereo",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        codec,
    ]
    if codec == "libx264":
        cmd += [
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-x264-params",
            "bframes=0",
        ]
    cmd += ["-pix_fmt", "yuv420p", "-r", str(int(fps))]
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
        "30",
        "-keyint_min",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-sample_fmt",
        "fltp",
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
        default="9000k",
        help="target video bitrate",
    )
    parser.add_argument(
        "--maxrate",
        default="9000k",
        help="maximum video bitrate",
    )
    parser.add_argument(
        "--bufsize",
        default="18000k",
        help="encoder buffer size",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="skip Google Drive upload of the recording",
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

    detected_w, detected_h = get_camera_resolution(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detected_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detected_h)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or detected_w)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or detected_h)
    out_width, out_height = STREAM_WIDTH, STREAM_HEIGHT

    # FFmpeg 5.x introduced changes to the "scale" filter that broke the
    # previous use of ``force_original_aspect_ratio=cover``.  To maintain
    # compatibility with FFmpeg 4.4 we instead scale the frame while
    # preserving the aspect ratio and pad to the desired resolution.
    # YouTube expects a full 1080p frame.  Using padding caused the
    # stream to be reported as not filling the frame, so scale directly
    # to the output size.
    # Always scale to the full output resolution and set the display
    # aspect ratio so YouTube displays the feed without padding. Also
    # force a 30fps output so the stream maintains a consistent frame
    # rate when sent to FFmpeg.
    filter_str = "scale=1920:1080,setsar=1,setdar=16/9,fps=30"

    # Force a 30fps capture rate so FFmpeg receives frames at a
    # consistent realtime pace. Some cameras report incorrect FPS
    # values so we explicitly override to match the FFmpeg input rate.
    fps = STREAM_FPS
    cap.set(cv2.CAP_PROP_FPS, 30)


    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = STREAM_FPS


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
    cap.set(cv2.CAP_PROP_FPS, 30)
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

    # Delete recordings older than 7 days
    cutoff = time.time() - 7 * 24 * 3600
    for old_file in output_dir.glob("*.mp4"):
        try:
            if old_file.stat().st_mtime < cutoff:
                old_file.unlink()
                print(f"Deleted old file {old_file.name}")
        except Exception as exc:
            print(f"Failed to delete {old_file}: {exc}")

    retries = 0
    max_retries = 5
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
            print("Starting FFmpeg stream...")
            print("Running FFmpeg command:", cmd_str)
            lf.write("Running FFmpeg command: " + cmd_str + "\n")
            lf.write(f"Audio enabled: {'-c:a' in cmd}\n")
            lf.write(f"Video enabled: {'-c:v' in cmd}\n")
            stderr_buffer = deque(maxlen=50)
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                bufsize=10**8,
            )
            lf.write(f"FFmpeg PID: {process.pid}\n")
            print(
                f"Spawned FFmpeg PID {process.pid} with stdin pipe? {process.stdin is not None}"
            )
            # Give FFmpeg a moment to start and capture any immediate errors
            startup_timeout = 5
            start_time = time.time()
            while time.time() - start_time < startup_timeout:
                if process.poll() is not None:
                    break
                if process.stderr in select.select([process.stderr], [], [], 0.1)[0]:
                    line = process.stderr.readline()
                    if line:
                        text = line.decode("utf-8", errors="ignore")
                        stderr_buffer.append(text)
                        lf.write(text)
                        print("[FFMPEG]", text, end="")
                else:
                    time.sleep(0.1)
            if process.poll() is not None:
                print(f"FFmpeg failed to launch with code {process.returncode}")
                lf.write(f"FFmpeg exited early with code {process.returncode}\n")
                err_text = "".join(stderr_buffer)
                if err_text:
                    print(err_text)
                    lf.write(err_text)
                suggest_ffmpeg_exit_causes()
                return

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

            thread_out = threading.Thread(
                target=_reader, args=(process.stdout, lf), daemon=True
            )
            thread_err = threading.Thread(
                target=log_ffmpeg_stderr,
                args=(process.stderr, lf, stderr_buffer),
                daemon=True,
            )
            thread_out.start()
            thread_err.start()
            first_frame = True
            frame_count = 0
            frame_interval = 1.0 / STREAM_FPS
            last_frame_time = time.time()
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        lf.write("Dropped frame\n")
                        print("Dropped frame")
                        elapsed = time.time() - last_frame_time
                        # Temporarily disabled pacing for debugging potential over-throttling
                        # time.sleep(max(0, frame_interval - elapsed))
                        last_frame_time = time.time()
                        continue
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
                        if process.poll() is not None:
                            msg = f"FFmpeg exited with code {process.returncode}"
                            print(msg)
                            lf.write(msg + "\n")
                            err_text = "".join(stderr_buffer)
                            if err_text:
                                print(err_text)
                                lf.write(err_text)
                            suggest_ffmpeg_exit_causes()
                            break
                        if process.stdin.closed:
                            raise BrokenPipeError("FFmpeg stdin closed")
                        if pix_fmt == "rgb24":
                            frame_conv = cv2.cvtColor(
                                frame_resized, cv2.COLOR_BGR2RGB
                            )
                            process.stdin.write(frame_conv.astype(np.uint8).tobytes())
                        else:
                            process.stdin.write(frame_resized.astype(np.uint8).tobytes())
                        print("Sending frame at", time.time(), file=sys.stderr)
                        process.stdin.flush()
                        frame_count += 1
                        if frame_count % 30 == 0:
                            print(f"Sent {frame_count} frames to FFmpeg")
                    except BrokenPipeError:
                        msg = "FFmpeg stdin pipe already closed (BrokenPipeError)."
                        print(msg)
                        lf.write(msg + "\n")
                        suggest_ffmpeg_exit_causes()
                        break
                    except OSError as exc:
                        msg = f"FFmpeg pipe closed: {exc}. Aborting stream."
                        print(msg)
                        lf.write(msg + "\n")
                        suggest_ffmpeg_exit_causes()
                        break
                    elapsed = time.time() - last_frame_time
                    if elapsed > frame_interval * 1.5:
                        warn = f"Frame delay {elapsed:.3f}s"
                        print(warn)
                        lf.write(warn + "\n")
                    # Temporarily disable pacing during debugging
                    # time.sleep(max(0, frame_interval - elapsed))
                    last_frame_time = time.time()
            except KeyboardInterrupt:
                pass
            finally:
                if process.stdin:
                    if process.stdin.closed:
                        lf.write("stdin already closed before cleanup\n")
                    else:
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
            if not args.no_upload:
                folder_id = os.getenv("GDRIVE_FOLDER_ID")
                if folder_id:
                    upload_thread = threading.Thread(
                        target=upload_after_stream, args=(record_file, folder_id)
                    )
                    upload_thread.start()
                    upload_thread.join()
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
