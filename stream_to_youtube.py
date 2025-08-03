import cv2
import csv
import subprocess
import time
import sys
import os
import re
import threading
from collections import deque
from urllib.parse import urlparse
import argparse

import roster

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:
    canvas = None  # type: ignore

try:
    import pytesseract
except Exception:
    pytesseract = None  # type: ignore
try:
    import psutil
except Exception:
    psutil = None  # type: ignore
from datetime import datetime
from pathlib import Path

WIDTH = 1920
HEIGHT = 1080
FPS = 30
RTMP_URL = "rtmp://a.rtmp.youtube.com/live2/xcuz-3x1d-9y7v-ghec-2xmh"
BITRATE = "6000k"
BUFSIZE = "12000k"
TEST_MODE = "--test" in sys.argv

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 2
# Larger font scale for the scoreboard overlay
SB_FONT_SCALE = 1.2

# Static scoreboard ROIs (y1, y2, x1, x2)
# Adjust these values to match the on-screen scoreboard layout
CLOCK_ROI = (50, 100, 900, 1100)
HOME_ROI = (100, 150, 830, 930)
AWAY_ROI = (100, 150, 1090, 1190)

# Play count alert settings
HALFTIME_SECS = 20 * 60  # halftime at 20:00
FINAL_WARNING_SECS = 34 * 60  # 6:00 remaining in a 40 minute game
HALFTIME_MIN_PLAYS = 3
FINAL_MIN_PLAYS = 7


def validate_rtmp_url(url: str) -> bool:
    """Return True if the RTMP/RTMPS URL looks valid."""
    parsed = urlparse(url)
    return parsed.scheme in {"rtmp", "rtmps"} and bool(parsed.netloc) and bool(parsed.path)


def unique_path(path: Path) -> Path:
    """Return a unique path by appending a counter if the file exists."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def generate_compliance_report(
    play_counts: dict[str, int], timestamp: str, team_name: str = "Team"
) -> None:
    """Create a compliance summary CSV and optional PDF."""

    report_dir = Path("video") / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "compliance_report.csv"

    summary: list[dict[str, str | int]] = []
    for pid in sorted(play_counts.keys()):
        count = play_counts[pid]
        status_str = "Met" if count >= FINAL_MIN_PLAYS else "Below"
        summary.append(
            {
                "player": f"#{pid} {roster.get_player_name(int(pid))}",
                "plays": count,
                "status": f"{'✅' if status_str == 'Met' else '❌'} {status_str}",
            }
        )

    with open(csv_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Player", "Plays", "Status"])
        for row in summary:
            writer.writerow([row["player"], row["plays"], row["status"].split()[1]])

    if canvas is not None:
        pdf_path = report_dir / f"compliance_{timestamp[:8]}.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, f"Compliance Report - {team_name}")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 70, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        y = height - 100
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Player")
        c.drawString(150, y, "Plays")
        c.drawString(220, y, "Status")
        c.setFont("Helvetica", 12)
        y -= 20
        for row in summary:
            c.drawString(50, y, str(row["player"]))
            c.drawString(150, y, str(row["plays"]))
            c.drawString(220, y, row["status"])
            y -= 20
        c.save()

    print("\nCompliance Summary:")
    for row in summary:
        print(f"{row['player']} - {row['plays']} plays - {row['status']}")


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Open an MP4 writer, falling back to MJPG if needed."""
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    return writer


def draw_label(
    frame: "cv2.Mat",
    text: str,
    org: tuple[int, int],
    font_scale: float = FONT_SCALE,
) -> None:
    """Draw text with a black background rectangle for contrast."""
    (w, h), _ = cv2.getTextSize(text, FONT, font_scale, THICKNESS)
    x, y = org
    cv2.rectangle(frame, (x - 5, y - h - 5), (x + w + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (x, y),
        FONT,
        font_scale,
        (255, 255, 255),
        THICKNESS,
        cv2.LINE_AA,
    )


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
        sb_text = f"\u23F1 {clock}     \U0001F3E0 {home}  -  {away} \U0001F6EB"
        (sw, sh), _ = cv2.getTextSize(sb_text, FONT, SB_FONT_SCALE, THICKNESS)
        draw_label(frame, sb_text, (frame.shape[1] - sw - 10, 60), SB_FONT_SCALE)
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


def parse_clock(clock: str) -> int | None:
    """Return the clock time in seconds or None if invalid."""
    m = re.match(r"(\d{1,2}):(\d{2})", clock)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def _log_ffmpeg_errors(pipe) -> None:
    """Stream FFmpeg stderr output in real time."""
    for line in iter(pipe.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            print(f"[ffmpeg] {text}")
    pipe.close()


def launch_ffmpeg(
    width: int,
    height: int,
    fps: float,
    record_path: Path,
    rtmp_url: str,
    test_mode: bool = False,
) -> subprocess.Popen:
    """Start the FFmpeg subprocess for streaming and recording."""

    record_path.parent.mkdir(parents=True, exist_ok=True)

    outputs = [f"[f=mp4]{record_path}"]
    if not test_mode:
        if rtmp_url.startswith("rtmp://"):
            outputs.insert(0, f"[f=flv]{rtmp_url}")
        else:
            print(f"⚠️ Invalid RTMP URL: {rtmp_url}. Recording to MP4 only.")
    else:
        print("⚠️ Test mode: recording to MP4 only, skipping RTMP.")

    ffmpeg_command = [
        "ffmpeg",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "pipe:",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-b:v",
        "4500k",
        "-maxrate",
        "5000k",
        "-bufsize",
        "10000k",
        "-g",
        "60",
        "-r",
        str(int(fps)),
        "-threads",
        "2",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-shortest",
        "-f",
        "tee",
        "|".join(outputs),
    ]

    try:
        process = subprocess.Popen(
            ffmpeg_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if process.stderr is not None:
            threading.Thread(
                target=_log_ffmpeg_errors, args=(process.stderr,), daemon=True
            ).start()
        return process
    except FileNotFoundError:
        print("❌ ffmpeg not found. Please install FFmpeg.")
        return None


def measure_fps(cap: cv2.VideoCapture, frames: int = 60) -> float:
    """Measure camera FPS by grabbing a number of frames."""
    start = time.time()
    count = 0
    while count < frames:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    elapsed = time.time() - start
    return count / elapsed if elapsed > 0 else 0.0


def initialize_camera(
    index: int, width: int, height: int, fps: int
) -> tuple[cv2.VideoCapture, "cv2.Mat", int, int, float] | tuple[None, None, int, int, float]:
    """Initialize a camera at the given index and resolution."""

    print(f"🎥 Attempting camera index {index} at {width}x{height}@{fps}fps")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"❌ Unable to open camera index {index}")
        return None, None, 0, 0, 0.0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"❌ Failed to read frame from camera index {index}")
        cap.release()
        return None, None, 0, 0, 0.0

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    measured_fps = measure_fps(cap)
    print(
        f"✅ Camera index {index} initialized at {actual_width}x{actual_height} ({measured_fps:.1f} FPS)"
    )
    return cap, frame, actual_width, actual_height, measured_fps



def main() -> None:
    if not validate_rtmp_url(RTMP_URL):
        print(f"❌ Invalid RTMP URL: {RTMP_URL}")
        return

    parser = argparse.ArgumentParser(description="Stream and record game footage")
    parser.add_argument(
        "--filename",
        help="Base name for output files; timestamp and .mp4 will be appended",
    )
    args = parser.parse_args()

    global WIDTH, HEIGHT, FPS

    cap: cv2.VideoCapture | None = None
    test_frame: "cv2.Mat" | None = None
    cam_width, cam_height, cam_fps = WIDTH, HEIGHT, FPS
    for idx in range(3):
        cap, test_frame, cam_width, cam_height, cam_fps = initialize_camera(
            idx, WIDTH, HEIGHT, FPS
        )
        if cap and cam_width >= WIDTH and cam_height >= HEIGHT and cam_fps >= 25:
            break
        if cap:
            cap.release()
        cap, test_frame, cam_width, cam_height, cam_fps = initialize_camera(
            idx, 1280, 720, FPS
        )
        if cap:
            break
    if cap is None or test_frame is None:
        print("❌ Camera failed to initialize. Check the camera connection or device index.")
        return
    print("✅ Successfully captured initial frame:", test_frame.shape)

    WIDTH, HEIGHT = cam_width, cam_height
    FPS = 60 if cam_fps >= 50 else 30

    output_dir = Path("video")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = args.filename if args.filename else "game"
    record_file = unique_path(output_dir / f"{base_name}_{timestamp}.mp4")
    log_file = unique_path(output_dir / f"{base_name}_{timestamp}_play_log.csv")

    process = launch_ffmpeg(WIDTH, HEIGHT, FPS, record_file, RTMP_URL, TEST_MODE)
    if process is None:
        return

    log_fp = open(log_file, "w", newline="")
    log_writer = csv.writer(log_fp)
    log_writer.writerow(["timestamp", "player_id"])
    start = time.time()
    alert_log_file = unique_path(output_dir / f"{base_name}_{timestamp}_alerts.log")
    alert_fp = open(alert_log_file, "w", encoding="utf-8")
    play_counts: dict[str, int] = {}
    plays_since_check = 0
    last_check_time = start
    halftime_alerted: set[str] = set()
    final_alerted: set[str] = set()
    summary_generated = False
    cv2.namedWindow("Stream Preview", cv2.WINDOW_NORMAL)
    frame_interval = 1.0 / FPS
    frame_count = 0
    bytes_sent = 0
    failed_reads = 0
    last_log = start
    fps_start = start
    fps_counter = 0
    no_frame_secs = 0
    ffmpeg_error = False
    last_score_update = 0.0
    scoreboard = ("--:--", "0", "0")
    warned_shape = False
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

    sub_log_path = unique_path(output_dir / f"{base_name}_{timestamp}_substitution_log.csv")
    sub_log_fp = open(sub_log_path, "w", newline="")
    sub_log_writer = csv.writer(sub_log_fp)
    sub_log_writer.writerow(["timestamp", "type", "player_id", "message"])

    sub_state: dict[str, str] = {}
    last_sub_check_time = start
    prev_clock_secs: int | None = None
    game_half = 1

    def check_alerts() -> None:
        nonlocal plays_since_check, last_check_time
        now_check = time.time()
        if plays_since_check < 5 and now_check - last_check_time < 60:
            return
        elapsed_secs = int(now_check - start)
        if elapsed_secs >= HALFTIME_SECS:
            for pid, cnt in play_counts.items():
                if cnt < HALFTIME_MIN_PLAYS and pid not in halftime_alerted:
                    msg = (
                        f"[\u26A0\uFE0F ALERT] #{pid} {roster.get_player_name(int(pid))} "
                        f"has only {cnt} plays at halftime"
                    )
                    print(msg)
                    alert_fp.write(msg + "\n")
                    halftime_alerted.add(pid)
        if elapsed_secs >= FINAL_WARNING_SECS:
            remaining = max(HALFTIME_SECS * 2 - elapsed_secs, 0)
            mins, secs = divmod(remaining, 60)
            for pid, cnt in play_counts.items():
                if cnt < FINAL_MIN_PLAYS and pid not in final_alerted:
                    msg = (
                        f"[\U0001F6A8 FINAL WARNING] #{pid} {roster.get_player_name(int(pid))} "
                        f"has only {cnt} plays \u2014 {mins}:{secs:02d} remaining"
                    )
                    print(msg)
                    alert_fp.write(msg + "\n")
                    final_alerted.add(pid)
        plays_since_check = 0
        last_check_time = now_check
        alert_fp.flush()

    def check_substitutions(clock_str: str) -> None:
        nonlocal last_sub_check_time, prev_clock_secs, game_half
        now_sub = time.time()
        if now_sub - last_sub_check_time < 60:
            return
        clock_secs = parse_clock(clock_str)
        if clock_secs is None:
            return
        if prev_clock_secs is not None and clock_secs > prev_clock_secs + 60:
            game_half = 2
        prev_clock_secs = clock_secs
        remaining_secs = clock_secs + (HALFTIME_SECS if game_half == 1 else 0)
        for pid, cnt in play_counts.items():
            if cnt >= 7:
                if pid in sub_state:
                    sub_state.pop(pid)
                continue
            need = 7 - cnt
            new_state = None
            msg = ""
            if remaining_secs < need * 60:
                new_state = "warn"
                mins, secs = divmod(remaining_secs, 60)
                msg = (
                    f"[\u23F1 TIME WARNING] #{pid} {roster.get_player_name(int(pid))} "
                    f"unlikely to hit 7 plays \u2014 {mins}:{secs:02d} remaining"
                )
            elif cnt < 4:
                new_state = "sub"
                msg = (
                    f"[\U0001F45F SUB IN] #{pid} {roster.get_player_name(int(pid))} "
                    f"needs {need} more plays \u2014 recommend subbing now"
                )
            if new_state and sub_state.get(pid) != new_state:
                print(msg)
                ts = datetime.now().strftime("%H:%M:%S")
                sub_log_writer.writerow([ts, new_state, pid, msg])
                sub_log_fp.flush()
                sub_state[pid] = new_state
            elif new_state is None and pid in sub_state:
                sub_state.pop(pid)
        last_sub_check_time = now_sub


    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            now = time.time()
            if ret and frame is not None:
                fps_counter += 1
                failed_reads = 0
            else:
                failed_reads += 1
                print("\u26a0\ufe0f Invalid frame received", file=sys.stderr)
                if failed_reads >= 5:
                    print("Camera lost signal. Attempting to reopen...")
                    cap.release()
                    reopened = False
                    for idx in range(3):
                        cap, new_frame, cam_width, cam_height, cam_fps = initialize_camera(
                            idx, WIDTH, HEIGHT, FPS
                        )
                        if cap and cam_width >= WIDTH and cam_height >= HEIGHT and cam_fps >= 25:
                            reopened = True
                            break
                        if cap:
                            cap.release()
                        cap, new_frame, cam_width, cam_height, cam_fps = initialize_camera(
                            idx, 1280, 720, FPS
                        )
                        if cap:
                            reopened = True
                            break
                    if not reopened or cap is None or new_frame is None:
                        print("❌ Unable to reopen camera. Exiting.")
                        break
                    frame = new_frame
                    WIDTH, HEIGHT = cam_width, cam_height
                    FPS = 60 if cam_fps >= 50 else 30
                    frame_interval = 1.0 / FPS
                    failed_reads = 0
                    continue
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
                if not warned_shape:
                    print(
                        f"\u26a0\ufe0f Unexpected frame shape: {frame.shape} — resizing to (1920, 1080)"
                    )
                    warned_shape = True
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
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
                check_substitutions(clock)
                if not summary_generated and parse_clock(clock) == 0:
                    generate_compliance_report(
                        play_counts, timestamp, os.getenv("TEAM_NAME", "Team")
                    )
                    summary_generated = True

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
                    play_counts[char] = play_counts.get(char, 0) + 1
                    plays_since_check += 1
                    check_alerts()
            if process and process.stdin:
                frame_bytes = frame.tobytes()
                try:
                    process.stdin.write(frame_bytes)
                    bytes_sent += len(frame_bytes)
                except (BrokenPipeError, OSError) as exc:
                    print(f"[\u274C ERROR] FFmpeg pipe closed ({exc.__class__.__name__})")
                    print("\a", end="")
                    ffmpeg_error = True
                    if process.poll() is not None:
                        process.wait()
                    process = None
                    process.stdin.write(frame.tobytes())
                    process.stdin.flush()
                except BrokenPipeError:
                    print("[\u274C Streaming ended: BrokenPipeError]")
                    break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Streaming frame #{frame_count}")
            now = time.time()
            if now - last_log >= 5:
                elapsed = now - start
                hours, rem = divmod(int(elapsed), 3600)
                minutes, seconds = divmod(rem, 60)
                avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
                file_size = record_file.stat().st_size if record_file.exists() else 0
                bitrate = (file_size * 8 / elapsed / 1000) if elapsed > 0 else 0.0
                input_bitrate = (bytes_sent * 8 / elapsed / 1000) if elapsed > 0 else 0.0
                print(
                    f"[STREAM STATUS] \u23F1\ufe0f {minutes:02d}:{seconds:02d} | Frames Sent: {frame_count} | Avg FPS: {avg_fps:.2f} | In: {input_bitrate:.0f} kbps | Out: {bitrate:.0f} kbps",
                )
                expected_frames = elapsed * FPS
                dropped_frames = max(0, expected_frames - frame_count)
                drop_rate = (
                    (dropped_frames / expected_frames) * 100 if expected_frames > 0 else 0
                )
                usage_str = ""
                if psutil is not None:
                    try:
                        proc = psutil.Process(os.getpid())
                        mem_mb = proc.memory_info().rss / (1024 * 1024)
                        cpu_pct = psutil.cpu_percent(interval=None)
                        usage_str = f" | CPU: {cpu_pct:.1f}% | Mem: {mem_mb:.1f} MB"
                    except Exception:
                        usage_str = ""
                print(
                    f"[STREAM STATUS] \u23F1\ufe0f {hours:02d}:{minutes:02d}:{seconds:02d} | Frames Sent: {frame_count} | Avg FPS: {avg_fps:.2f} | Frame Drop: {drop_rate:.2f}%{usage_str}",
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
            check_alerts()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping stream...")
    finally:
        cap.release()
        if process and process.stdin:
            process.stdin.close()
        if process:
            process.wait()
        log_fp.close()
        drive_log_fp.close()
        sub_log_fp.close()
        alert_fp.close()
        cv2.destroyAllWindows()

        if not summary_generated:
            generate_compliance_report(
                play_counts, timestamp, os.getenv("TEAM_NAME", "Team")
            )

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
                    "title": f"{base_name}_{timestamp}",
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
                    f"Created folder {base_name}_{timestamp} -> {folder_link}"
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

            compliance_csv = Path("video") / "reports" / "compliance_report.csv"
            if compliance_csv.exists():
                compl_drive = drive.CreateFile(
                    {
                        "title": compliance_csv.name,
                        "parents": [{"id": game_folder_id}],
                        "mimeType": "text/csv",
                    }
                )
                compl_drive.SetContentFile(str(compliance_csv))
                compl_drive.Upload()
                compl_url = (
                    f"https://drive.google.com/file/d/{compl_drive['id']}/view"
                )
                print(f"Uploaded {compliance_csv.name} -> {compl_url}")

            pdf_file = Path("video") / "reports" / f"compliance_{timestamp[:8]}.pdf"
            if pdf_file.exists():
                pdf_drive = drive.CreateFile(
                    {"title": pdf_file.name, "parents": [{"id": game_folder_id}]}
                )
                pdf_drive.SetContentFile(str(pdf_file))
                pdf_drive.Upload()
                pdf_url = f"https://drive.google.com/file/d/{pdf_drive['id']}/view"
                print(f"Uploaded {pdf_file.name} -> {pdf_url}")
        except Exception as exc:  # pragma: no cover - network/auth
            print(f"Google Drive upload failed: {exc}")


if __name__ == "__main__":
    main()
