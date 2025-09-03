#!/usr/bin/env python3
"""Launch a single ffmpeg process for streaming and local recording.

The stream and local files share the same H.264/AAC encode via the tee
muxer.  Hardware encoders are auto‑detected with fallbacks and the command is
restarted with exponential backoff if ffmpeg exits non‑zero.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


def detect_encoder() -> str:
    """Return the best available H.264 encoder."""
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"], text=True
        )
    except Exception as exc:  # pragma: no cover - ffmpeg always available in CI
        logging.error("unable to list encoders: %s", exc)
        return "libx264"
    if "h264_nvmpi" in out:
        return "h264_nvmpi"
    if "h264_v4l2m2m" in out:
        return "h264_v4l2m2m"
    return "libx264"


def parse_bufsize(br: str) -> str:
    if br.endswith("M") and br[:-1].isdigit():
        return f"{int(br[:-1]) * 2}M"
    return br


def preflight(args: argparse.Namespace) -> bool:
    """Best‑effort checks before invoking ffmpeg."""
    ok = True
    cam = Path(args.cam_dev)
    if not cam.exists():
        logging.error("camera device %s not found", cam)
        ok = False
    elif shutil.which("v4l2-ctl"):
        try:
            out = subprocess.check_output(
                ["v4l2-ctl", "--list-formats-ext", "-d", str(cam)], text=True
            )
            logging.info("v4l2-ctl top modes:\n%s", "\n".join(out.splitlines()[:10]))
        except Exception as exc:  # pragma: no cover - diagnostics only
            logging.warning("v4l2-ctl failed: %s", exc)

    # Audio device check
    if args.audio_backend == "pulse":
        cmd = ["pactl", "list", "sources", "short"]
        name = args.pulse_dev
    else:
        cmd = ["arecord", "-L"]
        name = args.alsa_dev
    try:
        out = subprocess.check_output(cmd, text=True)
        if name not in out:
            logging.warning("audio device %s not found in %s output", name, cmd[0])
    except Exception as exc:  # pragma: no cover - diagnostic
        logging.warning("audio check failed: %s", exc)

    # Output directory writability
    out_dir = Path(args.out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        test = out_dir / ".write_test"
        test.write_text("ok")
        test.unlink()
    except Exception as exc:
        logging.error("output dir %s not writable: %s", out_dir, exc)
        ok = False
    return ok


def build_cmd(args: argparse.Namespace, encoder: str) -> tuple[list[str], str, list[str]]:
    """Return (command, audio_desc, outputs)."""
    gop = args.fps * 2

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        "-fflags",
        "+genpts+igndts+discardcorrupt",
        "-f",
        "v4l2",
        "-input_format",
        args.cam_input_format,
        "-framerate",
        str(args.fps),
        "-video_size",
        args.size,
        "-thread_queue_size",
        "1024",
        "-i",
        args.cam_dev,
    ]

    if args.audio_backend == "pulse":
        cmd += ["-f", "pulse", "-thread_queue_size", "1024", "-i", args.pulse_dev]
        audio_desc = f"{args.pulse_dev} (pulse)"
    else:
        cmd += ["-f", "alsa", "-thread_queue_size", "1024", "-i", args.alsa_dev]
        audio_desc = f"{args.alsa_dev} (alsa)"

    cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    cmd += ["-vf", "scale=in_range=pc:out_range=tv,format=yuv420p"]

    if encoder == "h264_nvmpi":
        cmd += ["-c:v", "h264_nvmpi", "-b:v", args.bitrate, "-g", str(gop)]
    elif encoder == "h264_v4l2m2m":
        cmd += ["-c:v", "h264_v4l2m2m", "-b:v", args.bitrate, "-g", str(gop)]
    else:
        buf = parse_bufsize(args.bitrate)
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            args.x264_preset,
            "-tune",
            "zerolatency",
            "-b:v",
            args.bitrate,
            "-maxrate",
            args.bitrate,
            "-bufsize",
            buf,
            "-g",
            str(gop),
        ]

    cmd += ["-c:a", "aac", "-b:a", args.audio_bitrate, "-ar", "48000", "-ac", "2"]

    outputs: list[str] = []
    if args.stream:
        outputs.append(
            f"[f=flv:onfail=ignore]{args.yt_url.rstrip('/')}/{args.stream_key}"
        )
    if args.segment_seconds > 0:
        outputs.append(
            f"[f=segment:segment_time={args.segment_seconds}:reset_timestamps=1:strftime=1]"
            f"{args.out_dir}/{args.basename}_%Y%m%d_%H%M%S.{args.record_format}"
        )
    elif args.record_format == "mp4":
        outputs.append(
            f"[f=mp4:movflags=+faststart]{args.out_dir}/{args.basename}.mp4"
        )
    else:
        outputs.append(f"[f=matroska]{args.out_dir}/{args.basename}.mkv")

    cmd += [
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_at_eof",
        "1",
        "-rw_timeout",
        "15000000",
        "-f",
        "tee",
        "|".join(outputs),
    ]
    return cmd, audio_desc, outputs


def run_cmd(cmd: list[str]) -> tuple[int, list[str]]:
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    lines: deque[str] = deque(maxlen=1000)
    assert proc.stderr is not None
    for line in proc.stderr:
        sys.stderr.write(line)
        lines.append(line)
    ret = proc.wait()
    return ret, list(lines)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser()
    p.add_argument("--stream", default="false", choices=["true", "false"])
    p.add_argument(
        "--record-format", default=os.getenv("RECORD_FORMAT", "mkv"), choices=["mkv", "mp4"]
    )
    p.add_argument(
        "--segment-seconds", type=int, default=int(os.getenv("SEGMENT_SECONDS", "0"))
    )
    p.add_argument("--size", default=os.getenv("CAM_SIZE", "1280x720"))
    p.add_argument("--fps", type=int, default=int(os.getenv("CAM_FPS", "30")))
    p.add_argument("--bitrate", default=os.getenv("VIDEO_BITRATE", "6M"))
    p.add_argument("--audio-bitrate", default=os.getenv("AUDIO_BITRATE", "160k"))
    p.add_argument(
        "--audio-backend",
        default=os.getenv("AUDIO_BACKEND", "alsa"),
        choices=["alsa", "pulse"],
    )
    p.add_argument("--cam-dev", default=os.getenv("CAM_DEV", "/dev/video0"))
    p.add_argument(
        "--cam-input-format", default=os.getenv("CAM_INPUT_FORMAT", "mjpeg")
    )
    p.add_argument("--alsa-dev", default=os.getenv("ALSA_DEV", "hw:1,0"))
    p.add_argument("--pulse-dev", default=os.getenv("PULSE_DEV", "default"))
    p.add_argument("--stream-key", default=os.getenv("STREAM_KEY"))
    p.add_argument(
        "--yt-url",
        default=os.getenv("YT_URL", "rtmps://a.rtmps.youtube.com/live2"),
    )
    p.add_argument("--out-dir", default=os.getenv("OUT_DIR", "video/raw"))
    p.add_argument("--basename", default=os.getenv("BASENAME"))
    p.add_argument("--x264-preset", default=os.getenv("X264_PRESET", "veryfast"))
    p.add_argument("--max-retries", type=int, default=5)
    args = p.parse_args()

    args.stream = args.stream == "true"
    if args.stream and not args.stream_key:
        p.error("--stream-key required when --stream true")
    if not args.basename:
        args.basename = f"game_{time.strftime('%Y%m%d_%H%M%S')}"

    if not preflight(args):
        return 1

    encoder = detect_encoder()
    logging.info("chosen encoder: %s", encoder)

    cmd, audio_desc, outputs = build_cmd(args, encoder)
    sanitized = cmd[:]
    if args.stream:
        sanitized[-1] = sanitized[-1].replace(
            args.stream_key, f"{args.stream_key[:4]}***"
        )
    logging.info("ffmpeg cmd: %s", " ".join(sanitized))
    logging.info(
        "video_in=%s audio_in=%s outputs=%s",
        args.cam_dev,
        audio_desc,
        "|".join(outputs),
    )

    retries = 0
    delay = 1
    while True:
        ret, lines = run_cmd(cmd)
        if ret == 0:
            return 0
        logging.error("ffmpeg exited with %s", ret)
        logging.error("last stderr lines:\n%s", "".join(lines[-50:]))
        retries += 1
        if retries >= args.max_retries:
            logging.error("giving up after %d retries", retries)
            return ret
        logging.info("restarting ffmpeg in %s sec", delay)
        time.sleep(delay)
        delay = min(delay * 2, 60)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

