#!/usr/bin/env python3
import os, sys, json, re, pathlib

try:
    import env_loader
    env_loader.load_env()
except Exception:
    pass

CFG_PATH = pathlib.Path("config/gameday.json")

def load_cfg():
    if CFG_PATH.exists():
        try:
            with open(CFG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as ex:
            print(f"[gameday] ERROR: invalid JSON in {CFG_PATH}: {ex}", file=sys.stderr)
            sys.exit(2)
    return {}

def coalesce(cfg_key, env_key, default=None):
    v = os.environ.get(env_key)
    if v is not None and v.strip():
        return v.strip()
    return (cfg.get(cfg_key) if (cfg_key in cfg and cfg[cfg_key]) else default)

def _valid_rtmp(u: str) -> bool:
    if not u or not isinstance(u, str):
        return False
    u = u.strip()
    if not (u.startswith("rtmps://") or u.startswith("rtmp://")):
        return False
    if "/live2/" not in u:
        return False
    if any(ch in u for ch in "<>"):
        return False
    return len(u.rsplit("/live2/", 1)[-1].strip()) > 0

cfg = load_cfg()

video_dev   = coalesce("video_dev",   "VIDEO_DEV",   "/dev/video0")
pulse_src   = coalesce("pulse_source","PULSE_DEV",   "alsa_input.platform-sound.analog-stereo")
video_size  = coalesce("video_size",  "VIDEO_SIZE",  "1280x720")
fps         = int(coalesce("fps",     "FPS",         "30"))
rtmp_url    = coalesce("rtmp_url",    "YOUTUBE_RTMP_URL", "")

ok = True
if not pathlib.Path(video_dev).exists():
    print(f"[gameday] WARN: video device missing: {video_dev}", file=sys.stderr)
if not _valid_rtmp(rtmp_url):
    print("[gameday] missing or invalid RTMP URL", file=sys.stderr)
    ok = False

print(f"[gameday] Launch -> video={video_dev} {video_size}@{fps} | pulse={pulse_src} | rtmp={'set' if bool(rtmp_url) else 'MISSING'}", file=sys.stderr)
if not ok:
    print("[gameday] failed to resolve config.", file=sys.stderr)
    sys.exit(2)

# IMPORTANT: stdout must be JSON only
out = {
    "video_dev": video_dev,
    "pulse_source": pulse_src,
    "video_size": video_size,
    "fps": fps,
    "rtmp_url": rtmp_url,
}
print(json.dumps(out, separators=(",", ":")))

