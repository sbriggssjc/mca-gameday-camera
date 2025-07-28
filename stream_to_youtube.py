import cv2
import csv
import subprocess
import time
import sys
import os
from datetime import datetime
from pathlib import Path

WIDTH = 1920
HEIGHT = 1080
FPS = 30
RTMP_URL = "rtmp://a.rtmp.youtube.com/live2/xcuz-3x1d-9y7v-ghec-2xmh"
BITRATE = "6000k"
BUFSIZE = "12000k"

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 2


def draw_label(frame: "cv2.Mat", text: str, org: tuple[int, int]) -> None:
    """Draw text with a black background rectangle for contrast."""
    (w, h), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    x, y = org
    cv2.rectangle(frame, (x - 5, y - h - 5), (x + w + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA)


def overlay_info(frame: "cv2.Mat", frame_count: int) -> None:
    """Overlay time, LIVE label, and frame counter on the frame."""
    time_str = datetime.now().strftime("%H:%M:%S")
    # LIVE label in top-left
    draw_label(frame, "LIVE", (10, 30))
    # Current time in top-right
    (tw, th), _ = cv2.getTextSize(time_str, FONT, FONT_SCALE, THICKNESS)
    draw_label(frame, time_str, (frame.shape[1] - tw - 10, 30))
    # Frame counter in bottom-left
    frame_text = f"Frame: {frame_count}"
    (fw, fh), _ = cv2.getTextSize(frame_text, FONT, FONT_SCALE, THICKNESS)
    draw_label(frame, frame_text, (10, frame.shape[0] - 10))


def launch_ffmpeg(width: int, height: int, record_path: Path) -> subprocess.Popen:
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
        "-ac",
        "2",
        "-f",
        "tee",
        f"[f=flv]{RTMP_URL}|[f=mp4]{record_path}",
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

    output_dir = Path("video")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_file = output_dir / f"game_{timestamp}.mp4"
    log_file = output_dir / f"game_{timestamp}_play_log.csv"
    process = launch_ffmpeg(WIDTH, HEIGHT, record_file)
    log_fp = open(log_file, "w", newline="")
    log_writer = csv.writer(log_fp)
    log_writer.writerow(["timestamp", "player_id"])
    cv2.namedWindow("Stream Preview", cv2.WINDOW_NORMAL)
    frame_interval = 1.0 / FPS
    frame_count = 0
    start = time.time()
    last_log = start

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
            overlay_info(frame, frame_count)
            cv2.imshow("Stream Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                if key in {ord('q'), 27}:
                    break
                char = chr(key).upper()
                if ('1' <= char <= '9') or ('A' <= char <= 'Z'):
                    elapsed = int(time.time() - start)
                    minutes, seconds = divmod(elapsed, 60)
                    ts = f"{minutes:02d}:{seconds:02d}"
                    log_writer.writerow([ts, char])
                    log_fp.flush()
                    print(f"[LOG] Player {char} logged at {ts}")
            try:
                process.stdin.write(frame.tobytes())
                process.stdin.flush()
            except BrokenPipeError:
                print("FFmpeg pipe closed (BrokenPipeError). Exiting.")
                break

            frame_count += 1
            now = time.time()
            if now - last_log >= 5:
                elapsed = now - start
                minutes, seconds = divmod(int(elapsed), 60)
                avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(
                    f"[STREAM STATUS] \u23F1\ufe0f {minutes:02d}:{seconds:02d} | Frames Sent: {frame_count} | Avg FPS: {avg_fps:.2f}",
                    flush=True,
                )
                last_log = now

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
        log_fp.close()
        cv2.destroyAllWindows()

    # Upload the recorded file to Google Drive after streaming finishes
    drive_folder_id = os.getenv("GDRIVE_FOLDER_ID")
    if drive_folder_id:
        try:
            from pydrive.auth import GoogleAuth
            from pydrive.drive import GoogleDrive

            token_file = "token.json"
            gauth = GoogleAuth()
            gauth.LoadCredentialsFile(token_file)
            if gauth.credentials is None:
                gauth.LocalWebserverAuth()
            elif gauth.access_token_expired:
                gauth.Refresh()
            gauth.SaveCredentialsFile(token_file)
            drive = GoogleDrive(gauth)

            gfile = drive.CreateFile({"title": record_file.name,
                                     "parents": [{"id": drive_folder_id}]})
            gfile.SetContentFile(str(record_file))
            gfile.Upload()
            view_url = f"https://drive.google.com/file/d/{gfile['id']}/view"
            print(f"Uploaded {record_file.name} -> {view_url}")
        except Exception as exc:  # pragma: no cover - network/auth
            print(f"Google Drive upload failed: {exc}")


if __name__ == "__main__":
    main()
