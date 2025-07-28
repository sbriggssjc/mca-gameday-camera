import cv2
import subprocess
import time
import sys

WIDTH = 1920
HEIGHT = 1080
FPS = 30
RTMP_URL = "rtmp://a.rtmp.youtube.com/live2/xcuz-3x1d-9y7v-ghec-2xmh"
BITRATE = "6000k"
BUFSIZE = "12000k"


def launch_ffmpeg(width: int, height: int) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-re",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(FPS),
        "-i", "-",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-vf", "scale=1920:1080,fps=30,setsar=1,setdar=16/9",
        "-pix_fmt", "yuv420p",
        "-b:v", BITRATE,
        "-maxrate", BITRATE,
        "-bufsize", BUFSIZE,
        "-g", "60",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-ac", "2",
        "-f", "flv",
        RTMP_URL,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        sys.exit("Unable to open camera")

    process = launch_ffmpeg(WIDTH, HEIGHT)
    frame_interval = 1.0 / FPS
    frame_count = 0
    start = time.time()

    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\u26a0\ufe0f Invalid frame received", file=sys.stderr)
                continue
            if frame.shape != (HEIGHT, WIDTH, 3):
                print(
                    f"\u26a0\ufe0f Unexpected frame shape {frame.shape}",
                    file=sys.stderr,
                )
                continue
            try:
                process.stdin.write(frame.tobytes())
                process.stdin.flush()
            except BrokenPipeError:
                print("FFmpeg pipe closed (BrokenPipeError). Exiting.")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start
                print(
                    f"Sent {frame_count} frames in {elapsed:.2f} seconds",
                    file=sys.stderr,
                    flush=True,
                )

            if process.poll() is not None:
                print(f"FFmpeg exited with code {process.returncode}")
                break

            delay = frame_interval - (time.time() - loop_start)
            if delay > 0:
                time.sleep(delay)
    finally:
        cap.release()
        if process.stdin:
            process.stdin.close()
        process.wait()


if __name__ == "__main__":
    main()
