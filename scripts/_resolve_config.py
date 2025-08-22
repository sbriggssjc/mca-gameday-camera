#!/usr/bin/env python3
import json, os, sys, subprocess

CFG_PATH = os.environ.get("GAMEDAY_CFG", "config/gameday.json")

def read_json(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def list_pulse_sources():
    try:
        out = subprocess.check_output(["pactl", "list", "sources", "short"], text=True)
        return [ln.split("\t",1)[1] for ln in out.strip().splitlines() if ln.strip()]
    except Exception:
        return []

def choose_pulse_source(desired):
    sources = list_pulse_sources()
    if desired and any(desired in s for s in sources):
        return desired
    # Prefer RØDE if present
    for s in sources:
        if "VideoMic_GO_II" in s or "RODE" in s or "R__DE" in s:
            return s.split("\t")[0] if "\t" in s else s
    # Fallback to the first non-monitor source
    for s in sources:
        if "monitor" not in s.lower():
            return s.split("\t")[0] if "\t" in s else s
    return None

def main():
    cfg = read_json(CFG_PATH)
    # Pull from env with fallbacks
    rtmp = os.environ.get("YOUTUBE_RTMP_URL") or cfg.get("rtmp_url") or ""
    vdev = os.environ.get("VIDEO_DEV") or cfg.get("video_dev") or "/dev/video0"
    pdev = os.environ.get("PULSE_DEV") or cfg.get("pulse_source") or ""
    vsize = os.environ.get("VIDEO_SIZE") or cfg.get("video_size") or "1280x720"
    fps   = int(os.environ.get("FPS") or cfg.get("fps") or 30)

    # Auto-pick pulse if not set
    pdev = choose_pulse_source(pdev)

    ok = True
    if not rtmp: 
        print("[gameday] ERROR: No RTMP URL (YOUTUBE_RTMP_URL or config/gameday.json.rtmp_url).", file=sys.stderr); ok=False
    if not pdev:
        print("[gameday] ERROR: No Pulse audio source found.", file=sys.stderr); ok=False
    if not os.path.exists(vdev):
        print(f"[gameday] ERROR: Video device {vdev} not found.", file=sys.stderr); ok=False

    print(f"[gameday] Launch summary -> video={vdev} {vsize}@{fps}fps | pulse={pdev} | rtmp={'set' if bool(rtmp) else 'MISSING'}")
    if not ok: sys.exit(2)

    out = {
        "rtmp_url": rtmp,
        "video_dev": vdev,
        "pulse_source": pdev,
        "video_size": vsize,
        "fps": fps
    }
    print(json.dumps(out))
if __name__ == "__main__":
    main()
