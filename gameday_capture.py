#!/usr/bin/env python3
"""Robust game-day capture utility.

This script records a local MP4 while optionally streaming to YouTube.
It tries multiple video pipelines, falling back until one succeeds. A
sidecar JSON manifest is written on completion.

Only the most essential functionality is implemented here.  The goal is
robustness over absolute feature parity with the shell scripts it
replaces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from env_loader import load_env
except Exception:  # pragma: no cover - env_loader always exists in repo
    def load_env(path: str = ".env") -> None:
        if Path(path).exists():
            for line in Path(path).read_text().splitlines():
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


LOG_DIR = Path("livestream_logs")
VIDEO_DIR = Path("video")
HEARTBEAT_INTERVAL = 30
EARLY_FAIL_SECONDS = 10


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def setup_logging(ts: str) -> Path:
    """Configure logging to console and timestamped file."""
    LOG_DIR.mkdir(exist_ok=True)
    logfile = LOG_DIR / f"terminal_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logfile),
        ],
    )
    return logfile


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(args: List[str]) -> subprocess.CompletedProcess:
    """Run a subprocess and return ``CompletedProcess``."""
    return subprocess.run(args, capture_output=True, text=True, check=False)


def check_ffmpeg_protocols() -> bool:
    """Return True if ffmpeg supports rtmp/rtmps."""
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-protocols"], text=True
        )
    except Exception:
        logging.error("ffmpeg not found")
        return False
    good = all(x in out for x in ("rtmp", "rtmps", "tls"))
    if not good:
        logging.warning("ffmpeg lacks rtmp/rtmps/tls; forcing local-only mode")
    return good


def probe_camera(video_dev: str) -> str:
    """Return preferred format ('h264' or 'mjpeg')."""
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--device", video_dev, "--list-formats-ext"],
            text=True,
        )
    except Exception as exc:
        logging.warning("v4l2-ctl failed: %s", exc)
        return "mjpeg"
    if "H264" in out.upper():
        return "h264"
    return "mjpeg"


def detect_audio(dev: Optional[str]) -> Tuple[str, Optional[str]]:
    """Return (mode, device). mode in {alsa,pulse,silent}."""
    if dev:
        if dev.startswith("hw:") or dev.startswith("plughw:"):
            return "alsa", dev
        return "pulse", dev
    # Try Pulse default
    try:
        src = subprocess.check_output(
            ["pactl", "get-default-source"], text=True
        ).strip()
        if src:
            return "pulse", src
    except Exception:
        pass
    logging.warning("No audio device found; using silent track")
    return "silent", None


def wait_for_video_device(path: str, timeout: int = 20) -> None:
    """Wait until ``path`` is free. Prints owner processes if busy."""
    start = time.time()
    delay = 1
    while time.time() - start < timeout:
        proc = subprocess.run(["fuser", path], capture_output=True, text=True)
        if proc.returncode != 0:
            return
        pids = proc.stdout.strip()
        if pids:
            logging.warning("%s busy by PIDs: %s", path, pids)
        time.sleep(delay)
        delay = min(delay * 2, 5)
    logging.error("Device %s still busy after %ss", path, timeout)


# ---------------------------------------------------------------------------
# Pipeline building
# ---------------------------------------------------------------------------

def build_ffmpeg_command(
    plan: str,
    video_dev: str,
    res: str,
    fps: int,
    audio: Tuple[str, Optional[str]],
    rtmp_url: Optional[str],
    local_file: Path,
    *,
    duration: Optional[int] = None,
    local_only: bool = False,
) -> List[str]:
    """Return an ffmpeg command for the requested pipeline.

    Parameters
    ----------
    plan: str
        One of ``"A"``, ``"B"``, ``"C"``.
    audio: tuple
        (mode, device) where mode is ``alsa``, ``pulse`` or ``silent``.
    local_only: bool
        When ``True`` the flv leg is omitted.
    """

    base = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-fflags",
        "+genpts",
        "-start_at_zero",
        "-vsync",
        "1",
    ]

    if duration:
        base += ["-t", str(duration)]

    # Video input
    if plan == "A":
        base += [
            "-f",
            "v4l2",
            "-input_format",
            "h264",
            "-framerate",
            str(fps),
            "-video_size",
            res,
            "-i",
            video_dev,
        ]
        vcodec = ["-c:v", "copy"]
    else:  # plans B and C use MJPEG input
        base += [
            "-f",
            "v4l2",
            "-input_format",
            "mjpeg",
            "-framerate",
            str(fps),
            "-video_size",
            res,
            "-i",
            video_dev,
        ]
        if plan == "B":
            vcodec = [
                "-c:v",
                "h264_v4l2m2m",
                "-b:v",
                "3500k",
                "-maxrate",
                "4000k",
                "-bufsize",
                "6000k",
                "-g",
                "60",
                "-pix_fmt",
                "yuv420p",
            ]
        else:  # plan C
            vcodec = [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-tune",
                "zerolatency",
                "-b:v",
                "3500k",
                "-maxrate",
                "4000k",
                "-bufsize",
                "6000k",
                "-g",
                "60",
                "-pix_fmt",
                "yuv420p",
            ]

    # Audio
    mode, dev = audio
    if mode == "alsa":
        base += ["-f", "alsa", "-ar", "48000", "-ac", "1", "-i", dev]
    elif mode == "pulse":
        base += ["-f", "pulse", "-ar", "48000", "-ac", "1", "-i", dev]
    else:  # silent
        base += ["-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000"]
    acodec = ["-c:a", "aac", "-b:a", "128k", "-ar", "48000"]

    maps = ["-map", "0:v:0", "-map", "1:a:0"]

    outputs: List[str]
    if local_only or not rtmp_url:
        outputs = [
            "-f",
            "mp4",
            "-movflags",
            "+frag_keyframe+empty_moov+faststart",
            str(local_file),
        ]
    else:
        tee = (
            f"[f=flv:onfail=ignore]{rtmp_url}|"
            f"[f=mp4:movflags=+frag_keyframe+empty_moov+faststart]{local_file}"
        )
        outputs = ["-f", "tee", tee]

    return base + maps + vcodec + acodec + outputs


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def heartbeat(path: Path, stop: threading.Event) -> None:
    while not stop.wait(HEARTBEAT_INTERVAL):
        if path.exists():
            logging.info("heartbeat: %s %d bytes", path.name, path.stat().st_size)


def run_ffmpeg(cmd: List[str], stop: threading.Event) -> int:
    logging.info("ffmpeg cmd: %s", shlex.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    def reader():
        for line in proc.stdout:
            logging.info("[ffmpeg] %s", line.rstrip())
    t = threading.Thread(target=reader, daemon=True)
    t.start()

    while proc.poll() is None and not stop.is_set():
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            continue
    if stop.is_set() and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    t.join()
    return proc.returncode or 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gameday capture utility")
    p.add_argument("--rtmp-url", dest="rtmp_url")
    p.add_argument("--video-dev", default="/dev/video0")
    p.add_argument("--audio-dev")
    p.add_argument("--res", default="1280x720")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration", type=int)
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--probe-only", action="store_true")
    return p.parse_args()


def main() -> None:
    load_env()
    args = parse_args()

    # Apply environment defaults
    rtmp_url = args.rtmp_url or os.environ.get("YOUTUBE_RTMP_URL")
    video_dev = args.video_dev or os.environ.get("VIDEO_DEV", "/dev/video0")
    audio_dev = args.audio_dev or os.environ.get("PULSE_DEV") or os.environ.get("AUDIO_DEV")
    res = args.res or os.environ.get("RES", "1280x720")
    fps = int(args.fps or os.environ.get("FPS", 30))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = setup_logging(ts)
    VIDEO_DIR.mkdir(exist_ok=True)
    local_file = VIDEO_DIR / f"game_{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"

    logging.info("log file: %s", logfile)

    if args.probe_only:
        probe_path = Path("/tmp/test_cam_probe.mkv")
        probe_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "v4l2",
            "-input_format",
            "h264",
            "-framerate",
            str(fps),
            "-video_size",
            res,
            "-i",
            video_dev,
            "-t",
            "10",
            "-c",
            "copy",
            str(probe_path),
        ]
        if run_cmd(probe_cmd).returncode != 0:
            probe_cmd[probe_cmd.index("h264")] = "mjpeg"
            probe_cmd[probe_cmd.index("copy")] = "h264_v4l2m2m"
            run_cmd(probe_cmd)
        logging.info("probe saved to %s", probe_path)
        return

    wait_for_video_device(video_dev)

    has_rtmp = check_ffmpeg_protocols()
    if args.local_only or not has_rtmp:
        rtmp_url = None

    audio = ("silent", None) if args.no_audio else detect_audio(audio_dev)

    preferred_format = probe_camera(video_dev)
    plans = ["A", "B", "C"] if preferred_format == "h264" else ["B", "C"]

    summary = (
        f"[gameday] Launch -> video={video_dev} {res}@{fps} | "
        f"audio={audio[1] if audio[1] else 'silent'} | "
        f"rtmps={'SET' if rtmp_url else 'MISSING'} | pipeline={plans[0]}"
    )
    print(summary)
    logging.info(summary)

    stop = threading.Event()

    def handle(sig, frame):  # pragma: no cover - signal handler
        logging.info("received signal %s", sig)
        stop.set()

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    chosen = None
    for plan in plans:
        cmd = build_ffmpeg_command(
            plan,
            video_dev,
            res,
            fps,
            audio,
            rtmp_url,
            local_file,
            duration=args.duration,
            local_only=rtmp_url is None,
        )
        hb_stop = threading.Event()
        hb_thread = threading.Thread(target=heartbeat, args=(local_file, hb_stop), daemon=True)
        hb_thread.start()
        start_time = time.time()
        rc = run_ffmpeg(cmd, stop)
        hb_stop.set()
        hb_thread.join()
        if rc == 0 or time.time() - start_time > EARLY_FAIL_SECONDS:
            chosen = plan
            break
        logging.warning("pipeline %s failed quickly; trying next", plan)
    if chosen is None:
        logging.error("all pipelines failed")
        return

    size = local_file.stat().st_size if local_file.exists() else 0
    logging.info("local file: %s size=%d bytes", local_file, size)
    print(str(local_file))

    manifest = {
        "start_time": ts,
        "end_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "chosen_pipeline": chosen,
        "ffmpeg_command": cmd,
        "filesize_bytes": size,
        "sha256": sha256sum(local_file) if local_file.exists() else None,
    }
    manifest_path = local_file.with_suffix(".json")
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(manifest_path)


if __name__ == "__main__":  # pragma: no cover
    main()
