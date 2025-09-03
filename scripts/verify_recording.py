#!/usr/bin/env python3
"""Verify the newest local recording is H.264(yuv420p)+AAC."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def probe(path: Path, stream: str) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        stream,
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    data = subprocess.check_output(cmd, text=True)
    info = json.loads(data)["streams"][0]
    return info


def main() -> int:
    out_dir = Path(os.getenv("OUT_DIR", "video/raw"))
    files = sorted(out_dir.glob("*"), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"no recordings found in {out_dir}", file=sys.stderr)
        return 1

    latest = files[-1]
    v = probe(latest, "v:0")
    a = probe(latest, "a:0")
    if v.get("codec_name") != "h264" or v.get("pix_fmt") != "yuv420p" or a.get("codec_name") != "aac":
        print(
            f"bad streams: video={v.get('codec_name')}/{v.get('pix_fmt')} audio={a.get('codec_name')}",
            file=sys.stderr,
        )
        return 1
    num, den = v.get("r_frame_rate", "0/1").split("/")
    fps = int(round(int(num) / int(den))) if den != "0" else 0
    print(
        f"OK: H.264(yuv420p)+AAC {v.get('width')}x{v.get('height')}p{fps}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

