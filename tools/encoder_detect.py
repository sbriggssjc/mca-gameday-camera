#!/usr/bin/env python3
import subprocess, re, sys, shlex

# Preference: Jetson V4L2 → NVENC → software x264
CANDIDATES = [
    ("h264_v4l2m2m", "-c:v h264_v4l2m2m"),
    ("h264_nvenc",   "-c:v h264_nvenc"),
    ("libx264",      "-c:v libx264"),
]


def has_encoder(name: str) -> bool:
    try:
        out = subprocess.check_output(["ffmpeg","-hide_banner","-encoders"], text=True, stderr=subprocess.STDOUT)
        return re.search(rf"\b{name}\b", out) is not None
    except Exception:
        return False


def sanity_probe(flag: str) -> bool:
    # 1-sec null encode to verify the encoder really opens
    cmd = f"ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=1280x720:rate=30 -t 1 {flag} -f null -"
    try:
        subprocess.check_call(shlex.split(cmd))
        return True
    except Exception:
        return False


def pick(preferred_csv: str|None=None) -> str:
    order = CANDIDATES[:]
    if preferred_csv:
        want = [x.strip() for x in preferred_csv.split(",") if x.strip()]
        order = [x for x in CANDIDATES if x[0] in want] + [x for x in CANDIDATES if x[0] not in want]
    for name, flag in order:
        if has_encoder(name) and sanity_probe(flag):
            return flag
    if sanity_probe("-c:v libx264"):
        return "-c:v libx264"
    raise SystemExit("No working H.264 encoder found (tried h264_v4l2m2m, h264_nvenc, libx264).")


if __name__ == "__main__":
    pref = sys.argv[1] if len(sys.argv) > 1 else None
    print(pick(pref))

