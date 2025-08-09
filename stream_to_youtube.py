import cv2
import subprocess
import time
import numpy as np
import os
import re
import threading
import queue
import socket
from collections import deque
from urllib.parse import urlparse
import argparse
import signal
import logging
import shlex

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
from ffmpeg_utils import build_ffmpeg_args, run_ffmpeg_command, detect_encoder
from config import StreamConfig, load_config

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

YOUTUBE_URL = os.environ.get("YT_RTMP_URL", "<PUT_YOUR_RTMPS_URL_HERE>")


def ffmpeg_has_encoder(name: str) -> bool:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        return name in out
    except Exception:
        return False


def pick_h264_encoder():
    # Prefer Jetson HW paths if present, else software
    candidates = [
        "h264_v4l2m2m",  # common on Jetson builds
        "h264_nvmpi",  # some Jetson ffmpeg builds
        "h264_omx",  # older Jetson/Nano builds
        "libx264",  # software fallback (always last)
    ]
    for enc in candidates:
        if ffmpeg_has_encoder(enc):
            return enc
    return "libx264"


def build_audio_filter():
    """
    Returns a low-latency audio filter chain:
      - volume: primary gain boost (in dB, default +8 dB)
      - highpass: remove rumble/wind (default 120 Hz)
      - acompressor: tame peaks (3:1 ratio, low threshold)
      - alimiter: hard ceiling to prevent clipping (-1.0 dBTP)
      - dynaudnorm (optional): gentle auto-normalization for speech/crowd

    Tunable via env vars:
      AUDIO_GAIN_DB   -> default 8    (use 6–12 for field mics)
      AUDIO_HIGHPASS  -> default 120  (Hz)
      AUDIO_MODE      -> "speech" | "crowd" | "off"
                         "speech": adds light dynaudnorm
                         "crowd":  stronger dynaudnorm
                         "off":    no dynaudnorm (lowest latency)
    """
    gain_db = float(os.environ.get("AUDIO_GAIN_DB", "8"))
    highpass_hz = int(os.environ.get("AUDIO_HIGHPASS", "120"))
    mode = os.environ.get("AUDIO_MODE", "speech").strip().lower()

    # Core filters: gain -> highpass -> compressor -> limiter
    chain = [
        f"volume={gain_db}dB",
        f"highpass=f={highpass_hz}",
        "acompressor=threshold=-24dB:ratio=3:attack=5:release=100:makeup=6",
        "alimiter=limit=-1.0dB"
    ]

    # Optional adaptive normalizer (single pass, live-safe)
    if mode == "speech":
        # Mild
        chain.append("dynaudnorm=f=250:g=5:n=1:p=0.9")
    elif mode == "crowd":
        # Stronger leveling for noisy environments
        chain.append("dynaudnorm=f=200:g=7:n=1:p=0.8")

    chain.append("aformat=channel_layouts=stereo,pan=stereo|c0=c0|c1=c0")
    return ",".join(chain)


def build_ffmpeg_cmd(
    encoder: str,
    rtmp_url: str,
    dev="/dev/video0",
    alsa="hw:1,0",
    width=1280,
    height=720,
    fps=30,
    video_bitrate="5000k",
    audio_bitrate="160k",
    input_format="mjpeg",
    use_wallclock_ts=True,
):
    # Input legs with large thread queues to avoid blocking
    video_in = [
        "-f",
        "v4l2",
        "-thread_queue_size",
        "2048",
        "-input_format",
        input_format,
        "-framerate",
        str(fps),
        "-video_size",
        f"{width}x{height}",
    ]
    if use_wallclock_ts:
        video_in += ["-use_wallclock_as_timestamps", "1"]
    video_in += ["-i", dev]
    audio_in = [
        "-f",
        "alsa",
        "-thread_queue_size",
        "2048",
        "-ac",
        "1",
        "-ar",
        "48000",
        "-i",
        alsa,
    ]

    # Filters: ensure yuv420p for YouTube, keep fps stable
    vf = f"scale={width}:{height}:flags=bicubic,format=yuv420p,fps={fps}"

    # Encoder-specific options
    v_opts = [
        "-c:v",
        encoder,
        "-b:v",
        video_bitrate,
        "-maxrate",
        video_bitrate,
        "-bufsize",
        "10M",
        "-g",
        str(fps * 2),
    ]
    if encoder == "libx264":
        # Only libx264 supports these:
        v_opts += [
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
        ]
    else:
        # Some Jetson builds reject -profile/-level on v4l2m2m – omit them
        v_opts += ["-pix_fmt", "yuv420p"]

    # NEW: audio filter chain + bump bitrate
    a_filter = build_audio_filter()
    # Optional: duplicate mono to stereo
    # a_filter = a_filter + ",aformat=channel_layouts=stereo,pan=stereo|c0=c0|c1=c0"
    a_opts = [
        "-af", a_filter,
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ar", "48000",
    ]

    common = [
        "-vsync", "1",
        "-fflags", "+genpts",
        "-flush_packets", "1",
        "-movflags", "+faststart",
        "-f", "flv",
    ]

    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
    ] + video_in + audio_in + ["-vf", vf] + v_opts + a_opts + common + [rtmp_url]


def run_with_retries(
    rtmp_url: str,
    dev: str = "/dev/video0",
    alsa: str = "hw:1,0",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    video_bitrate: str = "5000k",
    audio_bitrate: str = "160k",
):
    attempts = [
        {"encoder": pick_h264_encoder(), "input_format": "mjpeg", "use_ts": True},
        {"encoder": "libx264", "input_format": "mjpeg", "use_ts": True},
        {"encoder": "libx264", "input_format": "yuyv422", "use_ts": True},
        {"encoder": "libx264", "input_format": "yuyv422", "use_ts": False},
    ]

    for i, cfg in enumerate(attempts, 1):
        cmd = build_ffmpeg_cmd(
            cfg["encoder"],
            rtmp_url,
            dev=dev,
            alsa=alsa,
            width=width,
            height=height,
            fps=fps,
            video_bitrate=video_bitrate,
            audio_bitrate=audio_bitrate,
            input_format=cfg["input_format"],
            use_wallclock_ts=cfg["use_ts"],
        )
        print(
            f"[DEBUG] Attempt {i}/{len(attempts)} using encoder={cfg['encoder']} format={cfg['input_format']} wallclock_ts={cfg['use_ts']}"
        )
        print("FFmpeg command:", " ".join(shlex.quote(c) for c in cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        saw_v4l2_corrupt = False
        saw_pts_drop = False
        rc = None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                if "Dequeued v4l2 buffer contains corrupted data (0 bytes)" in line:
                    saw_v4l2_corrupt = True
                if "invalid dropping" in line and "PTS" in line:
                    saw_pts_drop = True
        finally:
            rc = proc.wait()

        if rc == 0 and not (saw_v4l2_corrupt or saw_pts_drop):
            print("[INFO] Stream ended cleanly.")
            return rc
        else:
            print(
                f"[WARN] Stream ended rc={rc}, v4l2_corrupt={saw_v4l2_corrupt}, pts_drop={saw_pts_drop}. Retrying…"
            )

    print("[ERROR] All attempts failed.")
    return 1

ERROR_KEYWORDS = (
    "input/output error",
    "could not connect to youtube",
    "broken pipe",
)


def _halve_bitrate(value: str) -> str:
    """Return half of a bitrate string like '4500k'."""
    try:
        num = int(re.findall(r"(\d+)", value)[0])
        return f"{max(num // 2, 1)}k"
    except Exception:
        return value


def _run_rtmp_test(url: str) -> None:
    """Run a short FFmpeg dry run to verify RTMP(S) connectivity."""
    test_cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-re",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=128x72:rate=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000",
        "-t",
        "2",
        "-rtmp_flags",
        "prefer_ipv4",
        "-f",
        "flv",
        url,
    ]
    rc, _, stderr = run_ffmpeg_command(test_cmd, timeout=10)
    if rc != 0:
        print("[RTMP TEST] Unable to reach RTMP(S) URL:")
        print(stderr)


def find_usb_microphone(default_device: str = "hw:1,0") -> str:
    """Return ALSA identifier for a USB/RØDE microphone if present.

    Parameters
    ----------
    default_device: str
        Fallback ALSA device string (e.g., "hw:1,0").
    """

    result = subprocess.run(
        ["arecord", "-l"], capture_output=True, text=True, timeout=5
    )
    matches = re.findall(r"card (\d+): ([^\[]+)\[([^\]]+)\], device (\d+):", result.stdout)
    for card, name, desc, device in matches:
        if "rode" in name.lower() or "usb" in desc.lower():
            return f"hw:{card},{device}"
    return default_device  # fallback to specific device


def check_audio_input(device: str) -> bool:
    """Return True if audio device produces a non-silent signal."""

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
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
        if result.returncode != 0:
            print(f"⚠️ ALSA device {device} not found")
            return False
        if not result.stdout:
            print(f"⚠️ No audio captured from ALSA device {device}")
            return False
        audio = np.frombuffer(result.stdout, dtype=np.int16)
        if np.max(np.abs(audio)) == 0:
            print(f"⚠️ Silence detected on ALSA device {device}")
            return False
        return True
    except Exception as e:
        print(f"⚠️ Audio check failed for device {device}: {e}")
        return False


def detect_volume_gain(device: str, target_db: float = -15.0) -> float:
    """Return gain (in dB) needed to reach target mean volume.

    Runs a short ffmpeg dry-run using the ``volumedetect`` filter to measure
    the mean volume of the provided ALSA device. If successful, the difference
    between ``target_db`` and the measured value is returned. On failure, a
    default gain of ``2.5`` dB is used.
    """

    cmd = [
        "ffmpeg",
        "-f",
        "alsa",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-t",
        "3",
        "-i",
        device,
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    try:
        rc, _, stderr = run_ffmpeg_command(cmd, timeout=15)
        match = re.search(r"mean_volume:\s*(-?\d+\.?\d*) dB", stderr)
        if rc == 0 and match:
            measured_db = float(match.group(1))
            gain = target_db - measured_db
            print(
                f"[AUDIO] mean volume: {measured_db:.1f} dBFS, target: {target_db:.1f} dBFS, applying gain: {gain:.1f} dB"
            )
            return gain
    except Exception as e:
        print(f"⚠️ Volume detection failed: {e}")

    default_gain = 2.5
    print(f"[AUDIO] Using default gain: {default_gain} dB")
    return default_gain


def ping_rtmp(url: str, timeout: int = 5) -> bool:
    """Return True if the RTMP(S) endpoint is reachable."""

    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "rtmps" else 1935)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError as e:
        print(f"[❌ ERROR] TCP connection to {host}:{port} failed: {e}")
        return False


def diagnose_rtmp_connection(host: str = "a.rtmps.youtube.com", port: int = 443) -> None:
    """Diagnose DNS and TCP connectivity to the RTMP(S) host."""

    try:
        ip = socket.gethostbyname(host)
        print(f"[DIAG] DNS resolved {host} -> {ip}")
        dns_ok = True
    except socket.gaierror as e:
        print(f"[DIAG] DNS resolution failed for {host}: {e}")
        dns_ok = False

    if dns_ok:
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"[DIAG] TCP connection to {host}:{port} succeeded")
        except OSError as e:
            print(f"[DIAG] TCP connection to {host}:{port} failed: {e}")
    else:
        print("[DIAG] Skipping TCP check due to DNS failure")


AUDIO_LEVEL_DB = 0.0
LAST_BITRATE = 0.0
LAST_FRAME_TIME = 0.0
AUDIO_OK = False
VIDEO_OK = False
IO_ERROR_COUNT = 0
RTMP_REACHABLE = False

MAX_RESTART_ATTEMPTS = 3
restart_attempts = 0


def abort_stream() -> None:
    """Print a summary report and exit."""

    summary = [
        f"Retries: {restart_attempts}",
        f"RTMP reachable: {RTMP_REACHABLE}",
        f"Last bitrate: {LAST_BITRATE} kbps",
        f"Audio OK: {AUDIO_OK} (level {AUDIO_LEVEL_DB:.1f} dBFS)",
        f"Video OK: {VIDEO_OK}",
    ]
    print("[🛑 HALT] FFmpeg aborted. Summary:")
    for line in summary:
        print(" - " + line)
    sys.exit(1)


def handle_ffmpeg_crash(process):
    """Log crash details, back off, and track restart attempts."""

    global restart_attempts, IO_ERROR_COUNT, RTMP_REACHABLE
    restart_attempts += 1
    stderr_output = ""
    exit_code = process.returncode if process else None
    if process and process.stderr:
        try:
            stderr_output = process.stderr.read().decode(errors="replace")
        except Exception:
            stderr_output = ""
    if exit_code is not None:
        print(f"[FFMPEG EXIT CODE] {exit_code}")
    if stderr_output:
        print(f"[FFMPEG STDERR] {stderr_output}")
        Path("logs").mkdir(exist_ok=True)
        with Path("logs/ffmpeg_last_error.txt").open("w", encoding="utf-8") as fp:
            fp.write(stderr_output)
        RTMP_REACHABLE = ping_rtmp("rtmps://a.rtmps.youtube.com/live2")
        lower = stderr_output.lower()
        if "input/output error" in lower:
            IO_ERROR_COUNT += 1
            if IO_ERROR_COUNT == 2:
                diagnose_rtmp_connection()
                if sys.stdin.isatty():
                    ans = input(
                        "Have you confirmed the stream key is valid and YouTube Live is enabled? (y/N): "
                    )
                    if ans.strip().lower() != "y":
                        abort_stream()
        if any(k in lower for k in ERROR_KEYWORDS):
            print("[🚨 STREAM FAILURE] FFmpeg died due to RTMP error. This is likely:")
            print("  - YouTube Live not actively listening for stream")
            print("  - Network blockage or dropped connection")
            print("  - Too low FPS or broken pipe (check resolution + CPU)")
            print("  - Stream key issue (though unlikely if previously working)")
            print(f"  RTMP URL: {mask_stream_url(RTMP_URL)}")
    if restart_attempts > MAX_RESTART_ATTEMPTS:
        abort_stream()
    base_delay = 5
    if stderr_output:
        lower = stderr_output.lower()
        if any(p in lower for p in [
            "connection refused",
            "no route to host",
            "name or service not known",
            "temporary failure in name resolution",
        ]):
            base_delay = 15
    backoff = min(60, base_delay * (2 ** (restart_attempts - 1)))
    print(f"[WAIT] Backing off for {backoff} seconds before retry...")
    time.sleep(backoff)


def start_ffmpeg_process(ffmpeg_command):
    """Starts and returns a new subprocess.Popen for the FFmpeg command."""

    try:
        return subprocess.Popen(
            ffmpeg_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except Exception as e:  # pragma: no cover - defensive
        print(f"[❌ ERROR] Failed to start FFmpeg: {e}")
        return None


def handle_ffmpeg_crash_old(process):
    """Log crash details, back off, and track restart attempts."""

    global restart_attempts
    restart_attempts += 1
    stderr_output = ""
    if process and process.stderr:
        try:
            stderr_output = process.stderr.read().decode(errors="replace")
        except Exception:
            stderr_output = ""
    if stderr_output:
        print(f"[FFMPEG STDERR] {stderr_output}")
        lower = stderr_output.lower()
        if any(k in lower for k in ERROR_KEYWORDS):
            print("[🚨 STREAM FAILURE] FFmpeg died due to RTMP error. This is likely:")
            print("  - YouTube Live not actively listening for stream")
            print("  - Network blockage or dropped connection")
            print("  - Too low FPS or broken pipe (check resolution + CPU)")
            print("  - Stream key issue (though unlikely if previously working)")
            print(f"  RTMP URL: {mask_stream_url(RTMP_URL)}")
    if restart_attempts > MAX_RESTART_ATTEMPTS:
        print("[🛑 ABORT] Too many FFmpeg failures.")
        sys.exit(1)
    backoff = min(30, 2 ** restart_attempts)
    print(f"[WAIT] Backing off for {backoff} seconds before retry...")
    time.sleep(backoff)


def is_ffmpeg_alive(process) -> bool:
    """Return True if the given FFmpeg process is running."""

    return process is not None and process.poll() is None


def monitor_audio_level(
    device: str, stop_event: threading.Event, threshold_db: float = -60.0
) -> None:
    """Sample the audio device every 10s and update ``AUDIO_LEVEL_DB``.

    A warning is logged and printed if the mean volume over the 10s window is
    below ``threshold_db``.  Each warning is appended to ``silence_log.txt``.
    """

    global AUDIO_LEVEL_DB
    log_path = Path("silence_log.txt")
    while not stop_event.is_set():
        cmd = [
            "ffmpeg",
            "-f",
            "alsa",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-i",
            device,
            "-t",
            "10",
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ]
        try:
            rc, _, stderr = run_ffmpeg_command(cmd, timeout=15)
            match = re.search(r"mean_volume:\s*(-?\d+\.?\d*) dB", stderr)
            if rc == 0 and match:
                AUDIO_LEVEL_DB = float(match.group(1))
            else:
                AUDIO_LEVEL_DB = -80.0

            if AUDIO_LEVEL_DB <= threshold_db:
                msg = (
                    f"⚠️ Microphone silence detected ({AUDIO_LEVEL_DB:.1f} dBFS)"
                )
                logging.warning(msg)
                print(msg, flush=True)
                with log_path.open("a") as fp:
                    fp.write(f"{datetime.now().isoformat()} {msg}\n")
        except Exception as e:
            AUDIO_LEVEL_DB = -80.0
            print(f"⚠️ Audio monitoring failed: {e}")

        # Loop repeats automatically after ffmpeg completes (~10s)
        stop_event.wait(0.1)


def system_monitor(stop_event: threading.Event) -> None:
    """Log CPU (and GPU if available) usage periodically."""

    if psutil is None:
        return
    while not stop_event.is_set():
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        msg = f"[SYSTEM] CPU: {cpu:.1f}% | MEM: {mem:.1f}%"
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                gpu_t = temps.get("gpu") or temps.get("GPU")
                if gpu_t:
                    msg += f" | GPU Temp: {gpu_t[0].current:.1f}C"
        except Exception:
            pass
        print(msg, flush=True)
        stop_event.wait(5)



# SETTINGS
CAMERA_INDEX = 0
WIDTH = 0
HEIGHT = 0
FPS = 0

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
    """Basic validation for RTMP(S) URLs."""
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
        if not str(pid).isdigit():
            continue
        count = play_counts[pid]
        status_str = "Met" if count >= FINAL_MIN_PLAYS else "Below"
        try:
            player_id = int(pid)
            player_name = roster.get_player_name(player_id)
        except (ValueError, TypeError):
            player_name = "UNKNOWN"
        report_entry = f"#{pid} {player_name}"
        summary.append(
            {
                "player": report_entry,
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
    frame: "cv2.Mat",
    frame_count: int,
    scoreboard: tuple[str, str, str] | None = None,
    audio_level: float | None = None,
) -> None:
    """Overlay time, LIVE label, frame counter, scoreboard, and audio meter."""
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
    if audio_level is not None:
        audio_text = f"Audio: {audio_level:.1f} dBFS"
        (aw, ah), _ = cv2.getTextSize(audio_text, FONT, FONT_SCALE, THICKNESS)
        draw_label(frame, audio_text, (frame.shape[1] - aw - 10, frame.shape[0] - 10))
        if audio_level <= -60:
            warn_text = "NO AUDIO"
            draw_label(frame, warn_text, (10, 60))


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


def _log_ffmpeg_errors(pipe, log_fp, buffer) -> None:
    """Stream FFmpeg stderr output in real time and monitor bitrate."""

    global LAST_BITRATE, IO_ERROR_COUNT
    zero_start: float | None = None
    status_path = Path("logs/stream_status.log")
    status_path.parent.mkdir(exist_ok=True)
    with status_path.open("a", encoding="utf-8") as status_fp:
        for line in iter(pipe.readline, b""):
            text = line.decode("utf-8", errors="replace").rstrip()
            if not text:
                continue
            buffer.append(text)
            print(f"[ffmpeg] {text}")
            log_fp.write(text + "\n")
            log_fp.flush()
            lower = text.lower()
            if "input/output error" in lower or ("alsa" in lower and "error" in lower):
                warn = f"[FFMPEG WARNING] {text}"
                print(warn)
                log_fp.write(warn + "\n")
                log_fp.flush()
                IO_ERROR_COUNT += 1
                if IO_ERROR_COUNT == 2:
                    diagnose_rtmp_connection()
            match = re.search(r"bitrate=\s*(\d+\.?\d*)kbits/s", text)
            if match:
                bitrate = float(match.group(1))
                LAST_BITRATE = bitrate
                status_fp.write(
                    f"{datetime.now().isoformat()} bitrate={bitrate} audio={AUDIO_LEVEL_DB:.1f}dB\n"
                )
                status_fp.flush()
                if bitrate == 0:
                    if zero_start is None:
                        zero_start = time.time()
                    elif time.time() - zero_start >= 10:
                        msg = "\033[5;31m[ALERT] Output bitrate 0 kbps!\033[0m"
                        if time.time() - LAST_FRAME_TIME < 5:
                            msg += " Frames captured but not sent."
                        print(msg)
                        status_fp.write(f"{datetime.now().isoformat()} {msg}\n")
                        status_fp.flush()
                else:
                    zero_start = None
    pipe.close()
    log_fp.close()




def launch_ffmpeg(
    mic_input: str,
    volume_gain_db: float,
    *,
    record_path: str | None,
    preset: str,
    bitrate: str,
    maxrate: str,
    bufsize: str,
    gop: int,
    keyint_min: int,
    force_ipv4: bool = False,
    retry: bool = True,
    diagnose_only: bool = False,
) -> subprocess.Popen | None:
    """Start an FFmpeg process configured for streaming with tuned settings.

    If ``retry`` is True and startup fails with common RTMP errors, the
    function retries once with a reduced bitrate and forced IPv4 reconnect.
    """

    width, height, fps = WIDTH, HEIGHT, FPS
    try:
        video_encoder = detect_encoder()
    except RuntimeError as e:
        print(e)
        return None
    print("[INFO] Streaming raw BGR → FFmpeg rawvideo → RTMP using", video_encoder)
    log_dir = Path("livestream_logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"ffmpeg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_fp = log_file.open("w", encoding="utf-8", errors="replace")

    extra = ["-fflags", "nobuffer", "-flush_packets", "1"]
    if not retry:
        extra += [
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_delay_max",
            "2",
        ]
    ffmpeg_command = build_ffmpeg_args(
        video_source="-",
        audio_device=mic_input,
        output_url=RTMP_URL,
        audio_gain_db=volume_gain_db,
        resolution=f"{width}x{height}",
        framerate=int(fps),
        video_codec=video_encoder,
        video_is_pipe=True,
        preset=preset,
        bitrate=bitrate,
        maxrate=maxrate,
        bufsize=bufsize,
        gop=gop,
        keyint_min=keyint_min,
        local_record=record_path,
        force_ipv4=force_ipv4,
        extra_args=extra,
        diagnose_only=diagnose_only,
    )

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

        stderr_lines: list[str] = []
        if process.stderr is not None:
            threading.Thread(
                target=_log_ffmpeg_errors,
                args=(process.stderr, log_fp, stderr_lines),
                daemon=True,
            ).start()

        start = time.time()
        while time.time() - start < 15:
            if process is None or process.poll() is not None:
                err = "\n".join(stderr_lines)
                lower_err = err.lower()
                if any(k in lower_err for k in ERROR_KEYWORDS):
                    print("[🚨 STREAM FAILURE] FFmpeg died due to RTMP error. This is likely:")
                    print("  - YouTube Live not actively listening for stream")
                    print("  - Network blockage or dropped connection")
                    print("  - Too low FPS or broken pipe (check resolution + CPU)")
                    print("  - Stream key issue (though unlikely if previously working)")
                    print(f"  RTMP URL: {mask_stream_url(RTMP_URL)}")
                    log_fp.write(err + "\n")
                    log_fp.flush()
                    if retry:
                        reduced = _halve_bitrate(bitrate)
                        reduced_max = _halve_bitrate(maxrate)
                        reduced_buf = _halve_bitrate(bufsize)
                        return launch_ffmpeg(
                            mic_input,
                            volume_gain_db,
                            record_path=record_path,
                            preset=preset,
                            bitrate=reduced,
                            maxrate=reduced_max,
                            bufsize=reduced_buf,
                            gop=gop,
                            keyint_min=keyint_min,
                            force_ipv4=True,
                            retry=False,
                            diagnose_only=diagnose_only,
                        )
                    _run_rtmp_test(RTMP_URL)
                else:
                    print(f"❌ FFmpeg exited early: {err}")
                    log_fp.write(err + "\n")
                log_fp.close()
                return None
            time.sleep(0.5)
        return process
    except FileNotFoundError:
        print("❌ ffmpeg not found. Please install FFmpeg.")
        log_fp.close()
        return None
    except Exception as e:
        print(f"[❌ ERROR] Failed to launch FFmpeg: {e}")
        log_fp.close()
        return None


def restart_ffmpeg(
    process: subprocess.Popen | None,
    mic_input: str,
    volume_gain_db: float,
    *,
    record_path: str | None,
    preset: str,
    bitrate: str,
    maxrate: str,
    bufsize: str,
    gop: int,
    keyint_min: int,
    force_ipv4: bool = False,
    diagnose_only: bool = False,
) -> subprocess.Popen | None:
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
        try:
            stderr_output = ""
            if process.stderr:
                stderr_output = process.stderr.read().decode(errors="replace")
            print(f"[FFMPEG EXIT CODE] {process.returncode}")
            if stderr_output:
                print(f"[FFMPEG STDERR] {stderr_output}")
                if "input/output error" in stderr_output.lower():
                    print(
                        "[🚫 RTMP ERROR] Could not connect to YouTube. Check network or stream key."
                    )
                    raise RuntimeError("RTMP failure")
        except Exception:
            pass

    return launch_ffmpeg(
        mic_input,
        volume_gain_db,
        record_path=record_path,
        preset=preset,
        bitrate=bitrate,
        maxrate=maxrate,
        bufsize=bufsize,
        gop=gop,
        keyint_min=keyint_min,
        force_ipv4=force_ipv4,
        diagnose_only=diagnose_only,
    )



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
    global AUDIO_OK, VIDEO_OK, LAST_FRAME_TIME, RTMP_REACHABLE
    parser = argparse.ArgumentParser(description="Stream and record game footage")
    parser.add_argument(
        "--filename",
        help="Base name for output files; timestamp and .mp4 will be appended",
    )
    parser.add_argument("--stream_key", default=None, help="RTMP(S) stream URL")
    parser.add_argument(
        "--mic_device",
        dest="mic_device",
        default=None,
        help="ALSA mic device",
    )
    parser.add_argument(
        "--gain_boost",
        dest="gain_boost",
        type=float,
        default=None,
        help="Audio gain in dB",
    )
    parser.add_argument("--resolution", default=None, help="Capture resolution WxH")
    parser.add_argument("--fps", type=int, default=None, help="Capture FPS")
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument("--bitrate", default=None, help="Target video bitrate")
    parser.add_argument("--maxrate", default=None, help="Max video bitrate")
    parser.add_argument("--bufsize", default=None, help="Encoder buffer size")
    parser.add_argument("--encoder", default=None, help="Video encoder")
    parser.add_argument("--preset", default=None, help="Encoder preset")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--model", default=None, help="Model path for classifiers")
    parser.add_argument("--gop", type=int, default=60, help="GOP size")
    parser.add_argument(
        "--keyint_min", type=int, default=30, help="Minimum GOP keyframe interval"
    )
    parser.add_argument("--record", action="store_true", help="Also record locally")
    parser.add_argument("--audio_meter", action="store_true", help="Overlay audio levels")
    parser.add_argument("--dry_run", action="store_true", help="Test devices only")
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Run FFmpeg without publishing to RTMP",
    )
    parser.add_argument("--train", action="store_true", help="Enable self-learning mode")
    parser.add_argument("--label", action="store_true", help="Enable label review mode")
    parser.add_argument(
        "--force-ipv4",
        dest="force_ipv4",
        action="store_true",
        help="Always append -rtmp_flags prefer_ipv4",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg: StreamConfig = load_config(args.config, args)

    if cfg.train:
        print("🧠 Self-learning mode enabled")
    if cfg.label:
        print("🔖 Label review mode enabled")
    if cfg.model:
        print(f"🪬 Model: {cfg.model}")

    if args.diagnose_only:
        print("[DIAG] FFmpeg will run without streaming to RTMP")

    train_dir = Path("training/live")
    label_dir = Path("training/review")
    if cfg.train:
        train_dir.mkdir(parents=True, exist_ok=True)
    if cfg.label:
        label_dir.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()

    def _handle_signal(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    stream_url = args.stream_key or cfg.stream_key or os.getenv("YOUTUBE_STREAM_KEY")

    if not stream_url:
        raise ValueError("❌ Stream URL is missing or invalid. Aborting stream.")

    print(f"[DEBUG] Using stream URL: {stream_url}")

    global RTMP_URL, RTMP_REACHABLE
    RTMP_REACHABLE = ping_rtmp("rtmps://a.rtmps.youtube.com/live2")
    if not RTMP_REACHABLE and not args.diagnose_only:
        print("❌ Preflight check failed: cannot reach YouTube RTMP server")
        return
    if not ping_rtmp(stream_url):
        print("⚠️ RTMP endpoint unreachable; streaming may fail")

    RTMP_URL = stream_url
    if not validate_rtmp_url(RTMP_URL):
        print(f"❌ Invalid RTMP URL: {RTMP_URL}")
        return

    print(f"📡 Streaming to: {mask_stream_url(RTMP_URL)}")

    # Stream directly with FFmpeg using the Jetson hardware encoder. This
    # bypasses the prior OpenCV capture loop and avoids Python-based frame
    # piping that caused low FPS on the device.
    run_with_retries(RTMP_URL, alsa=args.mic_device or "hw:1,0")
    return

    global WIDTH, HEIGHT, FPS
    try:
        width_str, height_str = cfg.resolution.lower().split("x")
        WIDTH, HEIGHT = int(width_str), int(height_str)
    except ValueError:
        print(f"❌ Invalid resolution format: {cfg.resolution}")
        return
    FPS = cfg.fps

    cap: cv2.VideoCapture | None = None

    print_available_cameras()

    start_idx = cfg.camera
    print(f"🎥 Trying camera index {start_idx} at {WIDTH}x{HEIGHT}")
    cap = initialize_camera(start_idx, WIDTH, HEIGHT, FPS)
    if cap is None:
        alt_idx = 1 - start_idx if start_idx in {0, 1} else 0
        if alt_idx != start_idx:
            print(f"🎥 Trying camera index {alt_idx} at {WIDTH}x{HEIGHT}")
            cap = initialize_camera(alt_idx, WIDTH, HEIGHT, FPS)

    if cap is None:
        print("🔍 Scanning /dev/video* paths")
        cap = initialize_camera_path(WIDTH, HEIGHT, FPS)

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

    global VIDEO_OK, LAST_FRAME_TIME
    VIDEO_OK = True
    LAST_FRAME_TIME = time.time()
    print("✅ Successfully captured initial frame:", test_frame.shape)

    # Apply requested resolution/FPS; rotate later if camera delivers portrait frames
    if test_frame.shape[0] > test_frame.shape[1]:
        print("🔄 Rotating input frames for landscape orientation")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if abs(actual_fps - FPS) > 0.1:
        print(f"⚠️ Camera FPS not locked at {FPS}, actual: {actual_fps:.2f}")
    else:
        print(f"✅ Camera FPS locked at {actual_fps:.2f}")

    mic_candidates = [
        find_usb_microphone(cfg.mic_device),
        cfg.mic_device,
        "plughw:1,0",
        "default",
    ]
    mic_input = None
    global AUDIO_OK
    AUDIO_OK = False
    for dev in mic_candidates:
        if check_audio_input(dev):
            mic_input = dev
            AUDIO_OK = True
            break
    if mic_input is None:
        print("⚠️ No working microphone found")
        mic_input = cfg.mic_device
    print(f"🎤 Using microphone: {mic_input}")
    volume_gain_db = cfg.gain_boost
    monitor_stop = threading.Event()
    if args.dry_run:
        print("[DRY RUN] Camera and microphone initialized successfully")
        cap.release()
        monitor_stop.set()
        return

    threading.Thread(target=system_monitor, args=(monitor_stop,), daemon=True).start()
    if args.audio_meter:
        threading.Thread(
            target=monitor_audio_level, args=(mic_input, monitor_stop), daemon=True
        ).start()

    output_dir = Path("video")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = args.filename if args.filename else "game"
    record_file = unique_path(output_dir / f"{base_name}_{timestamp}.mp4")
    log_file = unique_path(output_dir / f"{base_name}_{timestamp}_play_log.csv")
    record_path = str(record_file) if args.record else None

    ffmpeg_process = launch_ffmpeg(
        mic_input,
        volume_gain_db,
        record_path=record_path,
        preset=cfg.preset,
        bitrate=cfg.bitrate,
        maxrate=cfg.maxrate,
        bufsize=cfg.bufsize,
        gop=args.gop,
        keyint_min=args.keyint_min,
        force_ipv4=cfg.force_ipv4,
        diagnose_only=args.diagnose_only,
    )
    if ffmpeg_process is None:
        monitor_stop.set()
        return

    ffmpeg_command = (
        ffmpeg_process.args
        if isinstance(ffmpeg_process.args, list)
        else [ffmpeg_process.args]
    )

    frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=FPS * 2)
    bytes_sent = 0
    encode_stop = threading.Event()
    encode_thread = threading.Thread()

    def do_ffmpeg_restart() -> bool:
        nonlocal ffmpeg_process, encode_stop, encode_thread, ffmpeg_error, last_output_time, ffmpeg_command
        encode_stop.set()
        encode_thread.join(timeout=2)
        handle_ffmpeg_crash(ffmpeg_process)
        ffmpeg_process = start_ffmpeg_process(ffmpeg_command)
        if ffmpeg_process is None:
            return False
        ffmpeg_error = False
        last_output_time = time.time()
        encode_stop = threading.Event()
        encode_thread = threading.Thread(
            target=encoder_worker, args=(encode_stop,), daemon=True
        )
        encode_thread.start()
        return True

    def encoder_worker(stop_evt: threading.Event) -> None:
        nonlocal bytes_sent, ffmpeg_process, ffmpeg_command
        target_fps = FPS or 30
        frame_interval = 1.0 / target_fps
        frame_count_local = 0
        while not stop_evt.is_set():
            start_time = time.time()
            try:
                frm = frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            if not is_ffmpeg_alive(ffmpeg_process):
                print("[⛔ HALT] FFmpeg is not running. Skipping frame send.")
                handle_ffmpeg_crash(ffmpeg_process)
                ffmpeg_process = start_ffmpeg_process(ffmpeg_command)
                continue
            try:
                if ffmpeg_process.stdin:
                    if frm.shape[:2] != (HEIGHT, WIDTH):
                        frm = cv2.resize(frm, (WIDTH, HEIGHT))
                    frame_yuv = cv2.cvtColor(frm, cv2.COLOR_BGR2YUV_I420)
                    data = frame_yuv.tobytes()
                    ffmpeg_process.stdin.write(data)
                    ffmpeg_process.stdin.flush()
                    bytes_sent += len(data)
                    frame_count_local += 1
                    if frame_count_local % 100 == 0:
                        print(f"🟢 {frame_count_local} frames sent to FFmpeg")
            except BrokenPipeError:
                print("❌ FFmpeg pipe broken — exiting stream loop")
                stop_evt.set()
                break
            except Exception as e:
                print(f"[❌ ERROR] Write to FFmpeg failed: {e}")
                stop_evt.set()
                break
            elapsed = time.time() - start_time
            delay = frame_interval - elapsed
            if delay > 0:
                time.sleep(delay)

    encode_stop = threading.Event()
    encode_thread = threading.Thread(
        target=encoder_worker, args=(encode_stop,), daemon=True
    )
    encode_thread.start()

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
    consecutive_capture_failures = 0
    MAX_CAPTURE_FAILURES = 10
    MAX_RECONNECT_ATTEMPTS = 5
    last_log = start
    fps_start = start
    fps_counter = 0
    capture_fps = 0.0
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

    last_output_time = time.time()

    def check_alerts() -> None:
        nonlocal plays_since_check, last_check_time
        now_check = time.time()
        if plays_since_check < 5 and now_check - last_check_time < 60:
            return
        elapsed_secs = int(now_check - start)
        if elapsed_secs >= HALFTIME_SECS:
            for pid, cnt in play_counts.items():
                if not str(pid).isdigit():
                    continue
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
                if not str(pid).isdigit():
                    continue
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
            if not str(pid).isdigit():
                continue
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
            if ffmpeg_process is None or ffmpeg_process.poll() is not None:
                print("[❌ ERROR] FFmpeg dead. Halting frame sending.")
                if not do_ffmpeg_restart():
                    stop_event.set()
                    break
                continue
            loop_start = time.time()
            ret, frame = cap.read()
            if ret and frame is not None:
                LAST_FRAME_TIME = time.time()
                VIDEO_OK = True
            else:
                VIDEO_OK = False
                consecutive_capture_failures += 1
                logging.warning(
                    "Invalid frame received (%d/%d)",
                    consecutive_capture_failures,
                    MAX_CAPTURE_FAILURES,
                )
                if consecutive_capture_failures > MAX_CAPTURE_FAILURES:
                    print(
                        "[🛑 Camera Error] Too many consecutive frame read failures. Restarting camera and FFmpeg."
                    )
                    cap.release()
                    for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
                        logging.info(
                            "Attempting to reconnect camera (%d/%d)",
                            attempt,
                            MAX_RECONNECT_ATTEMPTS,
                        )
                        cap = initialize_camera(0, WIDTH, HEIGHT, FPS)
                        if cap is None:
                            cap = initialize_camera(1, WIDTH, HEIGHT, FPS)
                        if cap is not None:
                            logging.info("Camera reinitialized successfully")
                            break
                        logging.error(
                            "Camera reinitialization attempt %d failed", attempt
                        )
                        time.sleep(1)
                    else:
                        logging.critical(
                            "Max camera reconnect attempts exceeded; shutting down"
                        )
                        break
                    if not do_ffmpeg_restart():
                        stop_event.set()
                        break
                    consecutive_capture_failures = 0
                    continue
                time.sleep(1 / FPS)
                continue
            consecutive_capture_failures = 0
            fps_counter += 1
            now = time.time()
            if now - fps_start >= 1:
                capture_fps = fps_counter / (now - fps_start)
                fps_counter = 0
                fps_start = now
                print(f"[FPS] Capture: {capture_fps:.2f}")
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

            overlay_info(
                frame,
                frame_count,
                scoreboard,
                AUDIO_LEVEL_DB if args.audio_meter else None,
            )
            cv2.imshow("Stream Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                if key in {ord('q'), 27}:
                    break
                elif cfg.label and key == ord('0'):
                    fname = label_dir / f"label_{int(time.time())}.jpg"
                    cv2.imwrite(str(fname), frame)
                    print(f"[LABEL] Saved frame for review: {fname}")
                else:
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
                        if cfg.train:
                            fname = train_dir / f"{char}_{int(time.time())}.jpg"
                            cv2.imwrite(str(fname), frame)
            try:
                frame_queue.put(frame, timeout=1)
            except queue.Full:
                print("⚠️ Frame queue full - dropping frame")

            if cfg.train and frame_count % (FPS * 5) == 0:
                fname = train_dir / f"auto_{int(time.time())}.jpg"
                cv2.imwrite(str(fname), frame)

            if ffmpeg_process is None or ffmpeg_process.poll() is not None:
                print("[❌ ERROR] FFmpeg process is not running.")
                if not do_ffmpeg_restart():
                    stop_event.set()
                    break

            elapsed_loop = time.time() - loop_start
            time.sleep(max(0, (1 / FPS) - elapsed_loop))

            frame_count += 1
            if frame_count % (5 * FPS) == 0:
                print(
                    f"[CAPTURE DEBUG] {datetime.now().strftime('%H:%M:%S')} - Frame {frame_count}"
                )
            if frame_count % (2 * FPS) == 0:
                print(f"Streaming frame #{frame_count}")
            now = time.time()
            if now - last_log >= 5:
                elapsed = now - start
                hours, rem = divmod(int(elapsed), 3600)
                minutes, seconds = divmod(rem, 60)
                file_size = (
                    record_file.stat().st_size
                    if record_path and record_file.exists()
                    else 0
                )
                output_kbps = (file_size * 8 / elapsed / 1000) if elapsed > 0 else 0.0
                encoded_kbps = (bytes_sent * 8 / elapsed / 1000) if elapsed > 0 else 0.0
                print(
                    f"[STREAM DEBUG] Sent frame {frame_count}, Encoded: {encoded_kbps:.0f} kbps, Output: {output_kbps:.0f} kbps",
                )
                if output_kbps > 0:
                    last_output_time = now
                elif now - last_output_time >= 10:
                    print("[⚠️ ALERT] Stream output stalled.")
                    if not do_ffmpeg_restart():
                        stop_event.set()
                        break
                    last_output_time = now
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
                    f"[STREAM STATUS] \u23F1\ufe0f {hours:02d}:{minutes:02d}:{seconds:02d} | Frames Sent: {frame_count} | Capture FPS: {capture_fps:.2f} | Frame Drop: {drop_rate:.2f}%{usage_str}",
                    flush=True,
                )
                last_log = now

            if (ffmpeg_process is None or ffmpeg_process.poll() is not None) and not ffmpeg_error:
                if ffmpeg_process is None:
                    print("[❌ ERROR] FFmpeg process is not running.")
                else:
                    ffmpeg_process.wait()
                    stderr_output = ""
                    if ffmpeg_process.stderr:
                        try:
                            stderr_output = ffmpeg_process.stderr.read().decode(errors="replace")
                        except Exception:
                            pass
                    print(f"[FFMPEG EXIT CODE] {ffmpeg_process.returncode}")
                    if stderr_output:
                        print(f"[FFMPEG STDERR] {stderr_output}")
                    ffmpeg_process = None
                ffmpeg_error = True
            check_alerts()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping stream...")
    finally:
        monitor_stop.set()
        encode_stop.set()
        encode_thread.join(timeout=2)
        cap.release()
        if ffmpeg_process and ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        if ffmpeg_process:
            ffmpeg_process.kill()
            ffmpeg_process.wait()
            try:
                stderr_output = ""
                if ffmpeg_process.stderr:
                    stderr_output = ffmpeg_process.stderr.read().decode(errors="replace")
                print(f"[FFMPEG EXIT CODE] {ffmpeg_process.returncode}")
                if stderr_output:
                    print(f"[FFMPEG STDERR] {stderr_output}")
            except Exception:
                pass
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

