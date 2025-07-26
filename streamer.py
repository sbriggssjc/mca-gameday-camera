import cv2
import subprocess


def livestream(youtube_url: str, device_index: int = 0) -> None:
    """Stream the given camera device to YouTube via RTMP.

    Parameters
    ----------
    youtube_url : rtmp://a.rtmp.youtube.com/live2/xcuz-3x1d-9y7v-ghec-2xmh
        The RTMP URL provided by YouTube including the stream key.
    device_index : int, optional
        Index of the capture device to use, by default 0.
    """
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open capture device {device_index}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    command = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-f", "flv",
        youtube_url,
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE)

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
        process.wait()
