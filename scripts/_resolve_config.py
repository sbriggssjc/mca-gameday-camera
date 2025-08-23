#!/usr/bin/env python3
import os, sys, json, re, pathlib

CFG_PATH = pathlib.Path("config/gameday.json")

def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)

def load_cfg():
    if CFG_PATH.exists():
        try:
            with open(CFG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as ex:
            eprint(f"[gameday] ERROR: invalid JSON in {CFG_PATH}: {ex}")
            sys.exit(2)
    return {}

def coalesce(cfg_key, env_key, default=None):
    v = os.environ.get(env_key)
    if v is not None and v.strip():
        return v.strip()
    return (cfg.get(cfg_key) if (cfg_key in cfg and cfg[cfg_key]) else default)

def valid_rtmp(url: str) -> bool:
    if not url:
        return False
    if not (url.startswith("rtmp://") or url.startswith("rtmps://")):
        return False
    # basic YouTube key check: groups of [a-z0-9-], at least 10 chars
    return bool(re.search(r"/live2/[A-Za-z0-9\-]{10,}$", url))

cfg = load_cfg()

video_dev   = coalesce("video_dev",   "VIDEO_DEV",   "/dev/video0")
pulse_src   = coalesce("pulse_source","PULSE_DEV",   "alsa_input.platform-sound.analog-stereo")
video_size  = coalesce("video_size",  "VIDEO_SIZE",  "1280x720")
fps         = int(coalesce("fps",     "FPS",         "30"))
rtmp_url    = coalesce("rtmp_url",    "YOUTUBE_RTMP_URL", "")

ok = True
if not pathlib.Path(video_dev).exists():
    eprint(f"[gameday] WARN: video device missing: {video_dev}")
if not valid_rtmp(rtmp_url):
    eprint("[gameday] missing or invalid RTMP URL")
    ok = False

eprint(f"[gameday] Launch -> video={video_dev} {video_size}@{fps} | pulse={pulse_src} | rtmp={'set' if bool(rtmp_url) else 'MISSING'}")
if not ok:
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

