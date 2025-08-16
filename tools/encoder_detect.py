#!/usr/bin/env python3
import subprocess, re, sys, shlex

# Ordered by preference on Jetson/PC
CANDIDATES = [
    ("h264_v4l2m2m",  "-c:v h264_v4l2m2m"),
    ("h264_nvenc",    "-c:v h264_nvenc"),
    ("h264_nvmpi",    "-c:v h264_nvmpi"),
    ("libx264",       "-c:v libx264"),
]

def has_encoder(name: str) -> bool:
    try:
        out = subprocess.check_output(["ffmpeg","-hide_banner","-encoders"], text=True, stderr=subprocess.STDOUT)
        return bool(re.search(rf"\b{name}\b", out))
    except Exception:
        return False

def sanity_probe(enc_flag: str) -> bool:
    # Try a 1s null encode to verify encoder actually opens
    cmd = f"ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=1280x720:rate=30 -t 1 {enc_flag} -f null -"
    try:
        subprocess.check_call(shlex.split(cmd))
        return True
    except subprocess.CalledProcessError:
        return False

def pick_encoder(preferred_csv: str|None=None) -> str:
    order = CANDIDATES
    if preferred_csv:
        # Move any preferred encs to the front in given order if present in overall list
        names = [n.strip() for n in preferred_csv.split(",") if n.strip()]
        ordered = []
        base = dict(CANDIDATES)
        for n in names:
            if n in base:
                ordered.append((n, base[n]))
        for n,flag in CANDIDATES:
            if n not in [x[0] for x in ordered]:
                ordered.append((n, flag))
        order = ordered

    for name, flag in order:
        if has_encoder(name) and sanity_probe(flag):
            return flag
    # last ditch: software x264 even if not listed (some builds gate it)
    if sanity_probe("-c:v libx264"):
        return "-c:v libx264"
    raise SystemExit("No working H.264 encoder found (tried v4l2m2m, nvenc, nvmpi, libx264).")

if __name__ == "__main__":
    pref = sys.argv[1] if len(sys.argv) > 1 else None
    print(pick_encoder(pref))
