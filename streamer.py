import cv2
import numpy as np
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path


def ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg is not installed or not in PATH")
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


def livestream(youtube_url: str, device_index: int = 0) -> None:
    """Stream the given camera device to YouTube via RTMP.

    Parameters
    ----------
    youtube_url : rtmp://a.rtmp.youtube.com/live2/xcuz-3x1d-9y7v-ghec-2xmh
        The RTMP URL provided by YouTube including the stream key.
    device_index : int, optional
        Index of the capture device to use, by default 0.
    """
    ensure_ffmpeg()

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open capture device {device_index}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    out_width, out_height = 640, 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0
    print(f"Capture settings: {width}x{height} @ {fps}fps")

    command = [
        ensure_ffmpeg(),
        "-loglevel",
        "verbose",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{out_width}x{out_height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", select_codec(),
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-f", "flv",
        youtube_url,
    ]

    log_dir = Path("livestream_logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"streamer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    lf = log_file.open("w", encoding="utf-8", errors="ignore")
    print("Running FFmpeg command:", " ".join(command))
    process = subprocess.Popen(
        command,
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
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("Captured empty frame from camera")
            if frame.shape[2] == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
            elif frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise ValueError(
                    f"Unexpected frame shape: {frame.shape} or dtype: {frame.dtype}"
                )
            if first_frame:
                print("Frame shape:", frame.shape, "dtype:", frame.dtype)
                cv2.imwrite("debug_frame.jpg", frame)
                first_frame = False
            print(
                f"Frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}"
            )
            frame_resized = cv2.resize(frame, (out_width, out_height))
            cv2.imshow("Debug Preview", frame_resized)
            cv2.waitKey(1)
            print(f"Writing frame of shape {frame_resized.shape} to FFmpeg")
            process.stdin.write(frame_resized.astype(np.uint8).tobytes())
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Sent {frame_count} frames to FFmpeg")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
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
        lf.close()
        if ret != 0:
            print("FFmpeg exited with error:", ret)
