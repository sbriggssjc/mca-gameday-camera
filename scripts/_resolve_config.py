#!/usr/bin/env python3
import json, os, sys, subprocess, shlex, re, pathlib

CFG_PATH = os.environ.get("GAMEDAY_CFG", "config/gameday.json")

def read_json(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def list_pulse_sources():
    # Fallback to pactl; don’t crash if not available
    try:
        out = subprocess.check_output(["pactl", "list", "sources", "short"], text=True)
        # format: index <TAB> name <TAB> module <TAB> state
        names = []
        for ln in out.strip().splitlines():
            parts = ln.split("\t")
            if len(parts) >= 2: names.append(parts[1])
        return names
    except Exception:
        return []

def choose_pulse_source(desired):
    sources = list_pulse_sources()
    # Honor an explicit override even if we cannot query Pulse for sources
    if desired and (not sources or any(desired in s for s in sources)):
        return desired
    # Prefer RØDE if present
    for s in sources:
        sl = s.lower()
        if "videomic" in sl or "r__de" in sl or "rode" in sl or "go_ii" in sl:
            return s
    # First non-monitor source
    for s in sources:
        if "monitor" not in s.lower():
            return s
    return None

def main():
    cfg = read_json(CFG_PATH)
    rtmp = os.environ.get("YOUTUBE_RTMP_URL") or cfg.get("rtmp_url") or ""
    vdev = os.environ.get("VIDEO_DEV") or cfg.get("video_dev") or "/dev/video0"
    pdev = os.environ.get("PULSE_DEV") or cfg.get("pulse_source") or ""
    vsize = os.environ.get("VIDEO_SIZE") or cfg.get("video_size") or "1280x720"
    fps   = int(os.environ.get("FPS") or cfg.get("fps") or 30)

    pdev = choose_pulse_source(pdev)

    ok = True
    if not rtmp:
        print("[gameday] ERROR: No RTMP URL (YOUTUBE_RTMP_URL or config/gameday.json.rtmp_url).", file=sys.stderr); ok=False
    if not pdev:
        print("[gameday] ERROR: No Pulse audio source found.", file=sys.stderr); ok=False
    if not os.path.exists(vdev):
        print(f"[gameday] ERROR: Video device {vdev} not found.", file=sys.stderr); ok=False

    print(
        f"[gameday] Launch -> video={vdev} {vsize}@{fps} | pulse={pdev} | rtmp={'set' if bool(rtmp) else 'MISSING'}",
        file=sys.stderr,
    )
    if not ok:
        sys.exit(2)

    # stdout must contain **only** the compact JSON blob used by the launcher.
    print(
        json.dumps(
            {
                "rtmp_url": rtmp,
                "video_dev": vdev,
                "pulse_source": pdev,
                "video_size": vsize,
                "fps": fps,
            }
        )
    )

if __name__ == "__main__":
    main()
