import cv2
import subprocess
import time
import numpy as np
import os
import re
import threading
from collections import deque
from urllib.parse import urlparse
import argparse

import roster
import csv
import sys

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

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def find_usb_microphone(default_device: str = "hw:1,0") -> str:
    """Return ALSA identifier for a USB/RØDE microphone if present.

    Parameters
    ----------
    default_device: str
        Fallback ALSA device string (e.g., "hw:1,0").
    """

    result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
    matches = re.findall(r"card (\d+): ([^\[]+)\[([^\]]+)\], device (\d+):", result.stdout)
    for card, name, desc, device in matches:
        if "rode" in name.lower() or "usb" in desc.lower():
            return f"hw:{card},{device}"
    return default_device  # fallback to specific device


def check_audio_input(device: str) -> None:
    """Log a warning if the specified ALSA device produces only silence."""

    try:
        result = subprocess.run(
            [
                "arecord",
                "-D",
                device,
                "-d",
                "1",
                "-f",
                "S16_LE",
                "-r",
                "44100",
                "-c",
                "2",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if not result.stdout:
            print(f"⚠️ No audio captured from ALSA device {device}")
            return
        audio = np.frombuffer(result.stdout, dtype=np.int16)
        if np.max(np.abs(audio)) == 0:
            print(f"⚠️ Silence detected on ALSA device {device}")
    except Exception as e:
        print(f"⚠️ Audio check failed for device {device}: {e}")


def get_available_video_encoder() -> str:
    """Detect and return a supported H.264 encoder."""

    import subprocess

    try:
        encoders = subprocess.check_output(
            ["ffmpeg", "-encoders"], stderr=subprocess.STDOUT
        ).decode()
        if "h264_nvmpi" in encoders:
            print("[DEBUG] Using encoder: h264_nvmpi")
            return "h264_nvmpi"
        elif "libx264" in encoders:
            print("[DEBUG] Using encoder: libx264")
            return "libx264"
        else:
            raise RuntimeError(
                "❌ No compatible H.264 encoder found (tried h264_nvmpi and libx264)."
            )
    except Exception as e:  # pragma: no cover - defensive
        raise RuntimeError(f"❌ Failed to detect available encoder: {e}")


# SETTINGS
CAMERA_INDEX = 0
WIDTH = 1280
HEIGHT = 720
FPS = 30

# RTMP destination will be set after stream key validation in main()
RTMP_URL = ""


def unique_path(path: Path) -> Path:
    """Return a unique file path by appending a counter if needed."""
    counter = 1
    stem = path.stem
    suffix = path.suffix
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        counter += 1
    return candidate


# Overlay and alert settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
SB_FONT_SCALE = 0.5
THICKNESS = 1
CLOCK_ROI = (0, 0, 0, 0)
HOME_ROI = (0, 0, 0, 0)
AWAY_ROI = (0, 0, 0, 0)
HALFTIME_SECS = 720
HALFTIME_MIN_PLAYS = 4
FINAL_WARNING_SECS = 1440
FINAL_MIN_PLAYS = 7


def validate_rtmp_url(url: str) -> bool:
    """Basic validation for RTMP URLs."""
    parsed = urlparse(url)
    return parsed.scheme in {"rtmp", "rtmps"} and bool(parsed.netloc) and bool(parsed.path)


def mask_stream_url(url: str) -> str:
    """Return the stream URL with the secret key portion hidden."""
    return re.sub(r"/[^/]+$", "/<hidden>", url)


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


def preprocess_frame(frame: "cv2.Mat") -> "cv2.Mat":
    """Rotate portrait frames and resize to the target resolution."""
    if frame.shape[0] > frame.shape[1]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if frame.shape[:2] != (HEIGHT, WIDTH):
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
    return frame


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


def _log_ffmpeg_errors(pipe, log_fp) -> None:
    """Stream FFmpeg stderr output in real time and monitor bitrate."""
    zero_count = 0
    for line in iter(pipe.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        if not text:
            continue
        print(f"[ffmpeg] {text}")
        log_fp.write(text + "\n")
        log_fp.flush()
        match = re.search(r"bitrate=\s*(\d+\.?\d*)kbits/s", text)
        if match:
            bitrate = float(match.group(1))
            if bitrate == 0:
                zero_count += 1
                if zero_count >= 3:
                    warn = "[FFMPEG WARNING] Output bitrate 0 kbps - stream might have stalled"
                    print(warn)
                    log_fp.write(warn + "\n")
                    log_fp.flush()
            else:
                zero_count = 0
    pipe.close()
    log_fp.close()


def launch_ffmpeg(mic_input: str) -> subprocess.Popen | None:
    """Start an FFmpeg process configured for 720p/30fps streaming."""

    width, height, fps = WIDTH, HEIGHT, FPS
    try:
        video_encoder = get_available_video_encoder()
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return None
    if video_encoder not in {"h264_nvmpi", "libx264"}:
        print(f"❌ Unsupported encoder: {video_encoder}")
        return None
    print("[SELECTED ENCODER]", video_encoder)
    log_dir = Path("livestream_logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"ffmpeg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_fp = log_file.open("w", encoding="utf-8", errors="replace")
    ffmpeg_command = [
        "ffmpeg",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-thread_queue_size",
        "512",
        "-f",
        "alsa",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-i",
        mic_input,
        "-c:v",
        video_encoder,
        "-pix_fmt",
        "yuv420p",
        "-af",
        "volume=1.0",
    ]

    if video_encoder == "libx264":
        ffmpeg_command += ["-preset", "veryfast", "-tune", "zerolatency"]

    ffmpeg_command += [
        "-b:v",
        "4500k",
        "-maxrate",
        "6000k",
        "-bufsize",
        "6000k",
        "-g",
        "60",
        "-keyint_min",
        "30",
        "-r",
        str(fps),
        "-fflags",
        "nobuffer",
        "-flush_packets",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-f",
        "flv",
        RTMP_URL,
    ]

    print("[FFMPEG COMMAND]", " ".join(ffmpeg_command))
    log_fp.write("FFMPEG COMMAND: " + " ".join(ffmpeg_command) + "\n")
    log_fp.flush()

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
                target=_log_ffmpeg_errors, args=(process.stderr, log_fp), daemon=True
            ).start()
        return process
    except FileNotFoundError:
        print("❌ ffmpeg not found. Please install FFmpeg.")
        log_fp.close()
        return None


def restart_ffmpeg(process: subprocess.Popen | None, mic_input: str) -> subprocess.Popen | None:
    """Restart the FFmpeg process if the stream stalls."""
    if process is not None:
        try:
            if process.stdin:
                process.stdin.close()
        except Exception:
            pass
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            pass
    return launch_ffmpeg(mic_input)


def initialize_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture | None:
    """Attempt to open a camera with the given settings.

    Returns the ``cv2.VideoCapture`` object if successful, otherwise ``None``.
    """

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    ret, _ = cap.read()
    if not ret:
        cap.release()
        return None

    return cap


def initialize_camera_path(width: int, height: int, fps: int) -> cv2.VideoCapture | None:
    """Attempt to open cameras using /dev/video* paths."""

    for idx in range(10):
        device_path = Path(f"/dev/video{idx}")
        if not device_path.exists():
            continue
        print(f"🎥 Trying device path {device_path}")
        cap = cv2.VideoCapture(str(device_path), cv2.CAP_V4L2)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        ret, _ = cap.read()
        if ret:
            return cap
        cap.release()
    return None


def print_available_cameras() -> None:
    """Print a list of available video devices for debugging."""

    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"], capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            print("📷 Available cameras:\n" + result.stdout)
            return
    except Exception:
        pass

    devices = sorted(str(p) for p in Path("/dev").glob("video*"))
    if devices:
        print("📷 Available /dev/video* devices: " + ", ".join(devices))
    else:
        print("❌ No /dev/video* devices found.")



def main() -> None:
    parser = argparse.ArgumentParser(description="Stream and record game footage")
    parser.add_argument(
        "--filename",
        help="Base name for output files; timestamp and .mp4 will be appended",
    )
    parser.add_argument(
        "--stream_key",
        type=str,
        help="YouTube stream key",
    )
    parser.add_argument(
        "--alsa_device",
        default="hw:1,0",
        help="ALSA audio input device (e.g., hw:1,0)",
    )
    args = parser.parse_args()

    stream_key = args.stream_key or os.getenv("YOUTUBE_STREAM_KEY")

    if not stream_key or "YOUR_STREAM_KEY" in stream_key:
        raise ValueError("❌ Stream key is missing or invalid. Aborting stream.")

    stream_url = f"rtmp://a.rtmp.youtube.com/live2/{stream_key}"
    print(f"[DEBUG] Using stream URL: {stream_url}")

    global RTMP_URL
    RTMP_URL = stream_url
    if not validate_rtmp_url(RTMP_URL):
        print(f"❌ Invalid RTMP URL: {RTMP_URL}")
        return

    print(f"📡 Streaming to: {mask_stream_url(RTMP_URL)}")

    global WIDTH, HEIGHT, FPS

    cap: cv2.VideoCapture | None = None

    print_available_cameras()

    print("🎥 Trying camera index 0 at 1920x1080")
    cap = initialize_camera(0, 1920, 1080, FPS)
    if cap is None:
        print("🎥 Trying camera index 1 at 1280x720")
        cap = initialize_camera(1, 1280, 720, FPS)

    if cap is None:
        print("🔍 Scanning /dev/video* paths")
        cap = initialize_camera_path(1280, 720, FPS)

    if cap is None:
        print("❌ Camera failed to initialize after scanning all paths.")
        print_available_cameras()
        return

    cap.set(cv2.CAP_PROP_FPS, FPS)
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Camera resolution: {cam_width}x{cam_height}")

    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("❌ Failed to read initial frame from camera.")
        cap.release()
        return

    print("✅ Successfully captured initial frame:", test_frame.shape)

    # Force 720p/30fps output; rotate later if camera delivers portrait frames
    if test_frame.shape[0] > test_frame.shape[1]:
        print("🔄 Rotating input frames for landscape orientation")
    WIDTH, HEIGHT, FPS = 1280, 720, 30
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if abs(actual_fps - FPS) > 0.1:
        print(f"⚠️ Camera FPS not locked at {FPS}, actual: {actual_fps:.2f}")
    else:
        print(f"✅ Camera FPS locked at {actual_fps:.2f}")

    mic_input = find_usb_microphone(args.alsa_device)
    print(f"🎤 Using microphone: {mic_input}")
    check_audio_input(mic_input)

    output_dir = Path("video")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = args.filename if args.filename else "game"
    record_file = unique_path(output_dir / f"{base_name}_{timestamp}.mp4")
    log_file = unique_path(output_dir / f"{base_name}_{timestamp}_play_log.csv")

    process = launch_ffmpeg(mic_input)
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
    frame_count = 0
    bytes_sent = 0
    failed_reads = 0
    last_log = start
    fps_start = start
    fps_counter = 0
    encode_counter = 0
    capture_fps = 0.0
    encode_fps = 0.0
    no_frame_secs = 0
    ffmpeg_error = False
    last_score_update = 0.0
    scoreboard = ("--:--", "0", "0")
    warned_shape = False
    highlight_dir = output_dir / "highlights"
    highlight_dir.mkdir(parents=True, exist_ok=True)
    buffer: deque = deque(maxlen=int(10 * FPS))
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

    out_zero_start: float | None = None
    out_zero_warned = False

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
            if not ret or frame is None:
                failed_reads += 1
                print("\u26a0\ufe0f Invalid frame received", file=sys.stderr)
                if failed_reads >= 3:
                    print("❌ cap.read() failed 3 times, shutting down.")
                    break
                time.sleep(1 / FPS)
                continue
            failed_reads = 0
            fps_counter += 1
            now = time.time()
            if now - fps_start >= 1:
                capture_fps = fps_counter / (now - fps_start)
                encode_fps = encode_counter / (now - fps_start)
                fps_counter = 0
                encode_counter = 0
                fps_start = now
                print(
                    f"[FPS] Capture: {capture_fps:.2f} | Encode: {encode_fps:.2f}"
                )
                if capture_fps == 0:
                    no_frame_secs += 1
                else:
                    no_frame_secs = 0
                if no_frame_secs >= 5:
                    print("[\u26A0\uFE0F ALERT] No frames received for 5 seconds. Stream may have stalled.")
                    print("\a", end="")
                    no_frame_secs = 5
            if frame.shape[0] > frame.shape[1] or frame.shape[:2] != (HEIGHT, WIDTH):
                if frame.shape[:2] != (HEIGHT, WIDTH) and not warned_shape:
                    print(
                        f"\u26a0\ufe0f Unexpected frame shape: {frame.shape} — resizing to ({WIDTH}, {HEIGHT})"
                    )
                    warned_shape = True
                frame = preprocess_frame(frame)
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
                    process.stdin.flush()
                    bytes_sent += len(frame_bytes)
                    encode_counter += 1
                    print(
                        f"[FFMPEG DEBUG] Wrote {len(frame_bytes)} bytes for frame {frame_count}",
                        flush=True,
                    )
                except (BrokenPipeError, IOError) as exc:
                    print(f"[\u274C ERROR] FFmpeg pipe closed ({exc.__class__.__name__})")
                    print("\a", end="")
                    ffmpeg_error = True
                    if process.poll() is not None:
                        process.wait()
                    process = None
                    break

            elapsed_loop = time.time() - loop_start
            time.sleep(max(0, (1 / FPS) - elapsed_loop))

            frame_count += 1
            if frame_count % (2 * FPS) == 0:
                print(f"Streaming frame #{frame_count}")
            now = time.time()
            if now - last_log >= 5:
                elapsed = now - start
                hours, rem = divmod(int(elapsed), 3600)
                minutes, seconds = divmod(rem, 60)
                file_size = record_file.stat().st_size if record_file.exists() else 0
                output_kbps = (file_size * 8 / elapsed / 1000) if elapsed > 0 else 0.0
                encoded_kbps = (bytes_sent * 8 / elapsed / 1000) if elapsed > 0 else 0.0
                print(
                    f"[STREAM DEBUG] Sent frame {frame_count}, Encoded: {encoded_kbps:.0f} kbps, Output: {output_kbps:.0f} kbps",
                )
                if output_kbps <= 0:
                    if out_zero_start is None:
                        out_zero_start = now
                    elif now - out_zero_start >= 5 and not out_zero_warned:
                        print("[\u26A0\uFE0F ALERT] Out: 0 kbps for over 5 seconds")
                        out_zero_warned = True
                    elif now - out_zero_start >= 10:
                        print("[\u26A0\uFE0F ALERT] Restarting FFmpeg due to stalled output")
                        process = restart_ffmpeg(process, mic_input)
                        out_zero_start = now
                else:
                    out_zero_start = None
                    out_zero_warned = False
                avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
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
                    f"[STREAM STATUS] \u23F1\ufe0f {hours:02d}:{minutes:02d}:{seconds:02d} | Frames Sent: {frame_count} | Capture FPS: {capture_fps:.2f} | Encode FPS: {encode_fps:.2f} | Frame Drop: {drop_rate:.2f}%{usage_str}",
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

