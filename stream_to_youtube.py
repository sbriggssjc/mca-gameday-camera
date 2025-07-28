import cv2
import csv
import subprocess
import time
import sys
import os
import re
from collections import deque

try:
    import pytesseract
except Exception:
    pytesseract = None  # type: ignore
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

# Static scoreboard ROIs (y1, y2, x1, x2)
# Adjust these values to match the on-screen scoreboard layout
CLOCK_ROI = (50, 100, 900, 1100)
HOME_ROI = (100, 150, 830, 930)
AWAY_ROI = (100, 150, 1090, 1190)


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Open an MP4 writer, falling back to MJPG if needed."""
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    return writer


def draw_label(frame: "cv2.Mat", text: str, org: tuple[int, int]) -> None:
    """Draw text with a black background rectangle for contrast."""
    (w, h), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    x, y = org
    cv2.rectangle(frame, (x - 5, y - h - 5), (x + w + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA)


def overlay_info(
    frame: "cv2.Mat", frame_count: int, scoreboard: tuple[str, str, str] | None = None
) -> None:
    """Overlay time, LIVE label, frame counter and scoreboard."""
    time_str = datetime.now().strftime("%H:%M:%S")
    # LIVE label in top-left
    draw_label(frame, "LIVE", (10, 30))
    # Current time in top-right
    (tw, th), _ = cv2.getTextSize(time_str, FONT, FONT_SCALE, THICKNESS)
    draw_label(frame, time_str, (frame.shape[1] - tw - 10, 30))
    if scoreboard:
        clock, home, away = scoreboard
        sb_text = f"\u23F1 {clock} | \U0001F3E0 {home} - {away} \U0001F6EB"
        (sw, sh), _ = cv2.getTextSize(sb_text, FONT, FONT_SCALE, THICKNESS)
        draw_label(frame, sb_text, (frame.shape[1] - sw - 10, 60))
    # Frame counter in bottom-left
    frame_text = f"Frame: {frame_count}"
    (fw, fh), _ = cv2.getTextSize(frame_text, FONT, FONT_SCALE, THICKNESS)
    draw_label(frame, frame_text, (10, frame.shape[0] - 10))


def extract_roi_text(roi: "cv2.Mat") -> str:
    """Return OCR text from the given ROI."""
    if pytesseract is None:
        return ""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(thresh, config="--psm 7")


def read_scoreboard(frame: "cv2.Mat") -> tuple[str, str, str]:
    """Read clock, home score and away score from the frame."""
    y1, y2, x1, x2 = CLOCK_ROI
    clock_roi = frame[y1:y2, x1:x2]
    clock_raw = extract_roi_text(clock_roi)
    clock = re.sub(r"[^0-9:]", "", clock_raw).strip()
    if len(clock) < 4:
        clock = "--:--"

    y1, y2, x1, x2 = HOME_ROI
    home_roi = frame[y1:y2, x1:x2]
    home_raw = extract_roi_text(home_roi)
    home = re.sub(r"\D", "", home_raw).strip() or "0"

    y1, y2, x1, x2 = AWAY_ROI
    away_roi = frame[y1:y2, x1:x2]
    away_raw = extract_roi_text(away_roi)
    away = re.sub(r"\D", "", away_raw).strip() or "0"

    return clock, home, away


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
    fps_start = start
    fps_counter = 0
    no_frame_secs = 0
    ffmpeg_error = False
    last_score_update = 0.0
    scoreboard = ("--:--", "0", "0")
    highlight_dir = output_dir / "highlights"
    highlight_dir.mkdir(parents=True, exist_ok=True)
    buffer: deque = deque(maxlen=int(30 * FPS))
    highlight_files: list[Path] = []
    prev_home = -1
    prev_away = -1
    drive_log_path = output_dir / "drive_summaries.csv"
    write_header = not drive_log_path.exists()
    drive_log_fp = open(drive_log_path, "a", newline="")
    drive_writer = csv.writer(drive_log_fp)
    if write_header:
        drive_writer.writerow(["start", "end", "home", "away", "duration", "clip"])
    drive_start_clock = "--:--"
    drive_start_time = time.time()


    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            now = time.time()
            if ret and frame is not None:
                fps_counter += 1
            else:
                print("\u26a0\ufe0f Invalid frame received", file=sys.stderr)
            if now - fps_start >= 1:
                fps_value = fps_counter / (now - fps_start)
                fps_counter = 0
                fps_start = now
                if fps_value == 0:
                    no_frame_secs += 1
                else:
                    no_frame_secs = 0
                if no_frame_secs >= 5:
                    print("[\u26A0\uFE0F ALERT] No frames received for 5 seconds. Stream may have stalled.")
                    print("\a", end="")
                    no_frame_secs = 5
            if not ret or frame is None:
                continue
            if frame.shape != (HEIGHT, WIDTH, 3):
                print(
                    f"\u26a0\ufe0f Unexpected frame shape {frame.shape}",
                    file=sys.stderr,
                )
                continue
            buffer.append(frame.copy())
            if time.time() - last_score_update >= 1.0:
                scoreboard = read_scoreboard(frame)
                last_score_update = time.time()
                clock, home_s, away_s = scoreboard
                try:
                    home_val = int(home_s)
                except ValueError:
                    home_val = prev_home
                try:
                    away_val = int(away_s)
                except ValueError:
                    away_val = prev_away
                if prev_home >= 0 and (
                    home_val > prev_home or away_val > prev_away
                ):
                    clip_name = (
                        f"highlight_{home_val}-{away_val}_"
                        f"{clock.replace(':', '-')}.mp4"
                    )
                    clip_path = highlight_dir / clip_name
                    duration = int(time.time() - drive_start_time)
                    overlay_text = (
                        f"{drive_start_clock} → {clock} | "
                        f"{prev_home}-{prev_away} → {home_val}-{away_val} | {duration}s"
                    )
                    writer = open_writer(clip_path, FPS, (WIDTH, HEIGHT))
                    for f in buffer:
                        frame_copy = f.copy()
                        cv2.putText(
                            frame_copy,
                            overlay_text,
                            (10, 30),
                            FONT,
                            FONT_SCALE,
                            (0, 255, 0),
                            THICKNESS,
                            cv2.LINE_AA,
                        )
                        writer.write(frame_copy)
                    writer.release()
                    drive_writer.writerow(
                        [
                            drive_start_clock,
                            clock,
                            str(home_val),
                            str(away_val),
                            f"{duration}s",
                            clip_name,
                        ]
                    )
                    drive_log_fp.flush()
                    print(
                        f"Drive {drive_start_clock}-{clock} {prev_home}-{prev_away} -> {home_val}-{away_val} ({duration}s)"
                    )
                    print(f"Saved highlight clip: {clip_path}")
                    highlight_files.append(clip_path)
                    drive_start_clock = clock
                    drive_start_time = time.time()
                elif prev_home < 0:
                    drive_start_clock = clock
                    drive_start_time = time.time()
                prev_home = home_val
                prev_away = away_val

            overlay_info(frame, frame_count, scoreboard)
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
            if process and process.stdin:
                try:
                    process.stdin.write(frame.tobytes())
                    process.stdin.flush()
                except BrokenPipeError:
                    print("[\u274C ERROR] FFmpeg pipe closed (BrokenPipeError)")
                    print("\a", end="")
                    ffmpeg_error = True
                    if process.poll() is not None:
                        process.wait()
                    process = None

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

            if process and process.poll() is not None and not ffmpeg_error:
                exit_code = process.poll()
                print(f"[\u274C ERROR] FFmpeg process exited unexpectedly with code {exit_code}")
                print("\a", end="")
                ffmpeg_error = True
                process.wait()
                process = None

            delay = frame_interval - (time.time() - loop_start)
            if delay > 0:
                time.sleep(delay)
    finally:
        cap.release()
        if process and process.stdin:
            process.stdin.close()
        if process:
            process.wait()
        log_fp.close()
        drive_log_fp.close()
        cv2.destroyAllWindows()

    # Upload the recorded file and log to Google Drive after streaming finishes
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

            use_subfolder = os.getenv("GDRIVE_USE_GAME_FOLDER")
            game_folder_id = drive_folder_id
            if use_subfolder:
                folder_meta = {
                    "title": f"game_{timestamp}",
                    "parents": [{"id": drive_folder_id}],
                    "mimeType": "application/vnd.google-apps.folder",
                }
                folder_file = drive.CreateFile(folder_meta)
                folder_file.Upload()
                game_folder_id = folder_file["id"]
                folder_link = (
                    f"https://drive.google.com/drive/folders/{folder_file['id']}"
                )
                print(
                    f"Created folder game_{timestamp} -> {folder_link}"
                )

            # upload video
            gfile = drive.CreateFile(
                {"title": record_file.name, "parents": [{"id": game_folder_id}]}
            )
            gfile.SetContentFile(str(record_file))
            gfile.Upload()
            view_url = f"https://drive.google.com/file/d/{gfile['id']}/view"
            print(f"Uploaded {record_file.name} -> {view_url}")

            # upload play log
            log_drive = drive.CreateFile(
                {
                    "title": log_file.name,
                    "parents": [{"id": game_folder_id}],
                    "mimeType": "text/csv",
                }
            )
            log_drive.SetContentFile(str(log_file))
            log_drive.Upload()
            log_view_url = (
                f"https://drive.google.com/file/d/{log_drive['id']}/view"
            )
            print(f"Uploaded {log_file.name} -> {log_view_url}")

            drive_summary_drive = drive.CreateFile(
                {
                    "title": drive_log_path.name,
                    "parents": [{"id": game_folder_id}],
                    "mimeType": "text/csv",
                }
            )
            drive_summary_drive.SetContentFile(str(drive_log_path))
            drive_summary_drive.Upload()
            drive_summary_url = (
                f"https://drive.google.com/file/d/{drive_summary_drive['id']}/view"
            )
            print(f"Uploaded {drive_log_path.name} -> {drive_summary_url}")

            for clip_path in highlight_files:
                clip_drive = drive.CreateFile(
                    {"title": clip_path.name, "parents": [{"id": game_folder_id}]}
                )
                clip_drive.SetContentFile(str(clip_path))
                clip_drive.Upload()
                clip_url = f"https://drive.google.com/file/d/{clip_drive['id']}/view"
                print(f"Uploaded {clip_path.name} -> {clip_url}")
        except Exception as exc:  # pragma: no cover - network/auth
            print(f"Google Drive upload failed: {exc}")


if __name__ == "__main__":
    main()
