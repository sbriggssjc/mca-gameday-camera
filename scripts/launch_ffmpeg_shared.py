#!/usr/bin/env python3
"""Wrapper for bin/launch_ffmpeg_shared.sh.
Parses CLI flags, runs preflight checks, launches ffmpeg with optional
watchdog restart logic.
"""
import argparse
import os
import subprocess
import sys
import time
import logging
from collections import deque
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import preflight  # scripts/preflight.py


def run_ffmpeg(env, args):
    cmd = ["bash", "bin/launch_ffmpeg_shared.sh"]
    cmd.extend([
        "--stream", str(args.stream).lower(),
        "--segment-seconds", str(args.segment_seconds),
        "--record-format", args.record_format,
        "--size", args.size,
        "--fps", str(args.fps),
        "--bitrate", args.bitrate,
    ])
    logging.info("launching ffmpeg")
    proc = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, text=True)
    last_lines = deque(maxlen=20)
    for line in proc.stderr:
        sys.stderr.write(line)
        last_lines.append(line)
    ret = proc.wait()
    if ret != 0:
        logging.error("ffmpeg exited with %s", ret)
        logging.error("Last 20 stderr lines:\n%s", "".join(last_lines))
    return ret


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", default="false", choices=["true", "false"])
    parser.add_argument("--segment-seconds", type=int, default=int(os.getenv("SEGMENT_SECONDS", 0)))
    parser.add_argument("--record-format", default=os.getenv("RECORD_FORMAT", "mkv"), choices=["mkv", "mp4"], dest="record_format")
    parser.add_argument("--size", default=os.getenv("CAM_SIZE", "1280x720"))
    parser.add_argument("--fps", type=int, default=int(os.getenv("CAM_FPS", 30)))
    parser.add_argument("--bitrate", default=os.getenv("VIDEO_BITRATE", "6M"))
    parser.add_argument("--stream-key", default=os.getenv("STREAM_KEY"))
    parser.add_argument("--yt-url", default=os.getenv("YT_URL", "rtmps://a.rtmps.youtube.com/live2"))
    parser.add_argument("--out-dir", default=os.getenv("OUT_DIR", "./video/raw"))
    parser.add_argument("--basename", default=os.getenv("BASENAME"))
    args = parser.parse_args()

    if args.stream == "true" and not args.stream_key:
        parser.error("--stream-key required when --stream true")

    env = os.environ.copy()
    env.update(
        STREAM="true" if args.stream == "true" else "false",
        SEGMENT_SECONDS=str(args.segment_seconds),
        RECORD_FORMAT=args.record_format,
        CAM_SIZE=args.size,
        CAM_FPS=str(args.fps),
        VIDEO_BITRATE=args.bitrate,
        STREAM_KEY=args.stream_key or "",
        YT_URL=args.yt_url,
        OUT_DIR=args.out_dir,
    )
    if args.basename:
        env["BASENAME"] = args.basename

    # Preflight checks
    if not preflight.run(env):
        return 1

    retries = 0
    delay = 1
    while True:
        ret = run_ffmpeg(env, args)
        if ret == 0:
            return 0
        retries += 1
        if retries >= 5:
            logging.error("ffmpeg failed after %d retries", retries)
            return ret
        logging.info("restarting ffmpeg in %s sec", delay)
        time.sleep(delay)
        delay = min(delay * 2, 60)


if __name__ == "__main__":
    raise SystemExit(main())
