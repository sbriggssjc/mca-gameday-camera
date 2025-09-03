#!/usr/bin/env python3
"""Verify newest local recording contains H.264 video and AAC audio."""
import subprocess
from pathlib import Path
import sys
import os

def ffprobe_field(path: Path, stream: str, field: str) -> str:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", stream,
        "-show_entries", f"stream={field}",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ]
    return subprocess.check_output(cmd, text=True).strip().split("\n")[0]

def main() -> int:
    out_dir = Path(os.getenv("OUT_DIR", "./video/raw"))
    files = sorted(out_dir.glob("*"), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"no recordings found in {out_dir}", file=sys.stderr)
        return 1
    latest = files[-1]
    vcodec = ffprobe_field(latest, "v:0", "codec_name")
    pix = ffprobe_field(latest, "v:0", "pix_fmt")
    acodec = ffprobe_field(latest, "a:0", "codec_name")
    print(f"{latest}: video={vcodec} pix_fmt={pix} audio={acodec}")
    if vcodec != "h264" or pix != "yuv420p" or acodec != "aac":
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
