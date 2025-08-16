#!/usr/bin/env python3
import subprocess, re, sys, shlex

CANDIDATES = [
    ("h264_v4l2m2m", "-c:v h264_v4l2m2m"),
    ("h264_nvenc",   "-c:v h264_nvenc"),
    ("libx264",      "-c:v libx264"),
]

def has_encoder(name):
    try:
        out = subprocess.check_output(["ffmpeg","-hide_banner","-encoders"], text=True, stderr=subprocess.STDOUT)
        return re.search(rf"\b{name}\b", out) is not None
    except Exception:
        return False

def probe(flag):
    cmd = f"ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=1280x720:rate=30 -t 1 {flag} -f null -"
    try:
        subprocess.check_call(shlex.split(cmd))
        return True
    except Exception:
        return False

def pick(pref=None):
    order = CANDIDATES[:]
    if pref:
        wanted = [x.strip() for x in pref.split(",") if x.strip()]
        order = [(n,f) for n,f in CANDIDATES if n in wanted] + [(n,f) for n,f in CANDIDATES if n not in wanted]
    for name,flag in order:
        if has_encoder(name) and probe(flag):
            return flag
    if probe("-c:v libx264"):
        return "-c:v libx264"
    raise SystemExit("No working H.264 encoder found.")

if __name__ == "__main__":
    pref = sys.argv[1] if len(sys.argv) > 1 else None
    print(pick(pref))
