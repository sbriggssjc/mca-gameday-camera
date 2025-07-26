import cv2
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    command = [
        ensure_ffmpeg(),
        "-loglevel",
        "verbose",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
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
    lf = log_file.open("w")
    print("Running FFmpeg command:", " ".join(command))
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _reader(pipe, logf):
        for line in pipe:
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            print(line, end="")
            logf.write(line)

    thread = threading.Thread(target=_reader, args=(process.stdout, lf), daemon=True)
    thread.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            process.stdin.write(frame.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if process.stdin:
            process.stdin.close()
        ret = process.wait()
        thread.join()
        lf.write(f"\nffmpeg exited with code {ret}\n")
        lf.close()
        if ret != 0:
            print("FFmpeg exited with error:", ret)
