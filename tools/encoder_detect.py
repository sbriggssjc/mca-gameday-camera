#!/usr/bin/env python3
import subprocess, re, sys, shlex, argparse

# Preference order
BASE_CANDIDATES = [
    ("h264_v4l2m2m", "-c:v h264_v4l2m2m"),
    ("h264_nvenc",   "-c:v h264_nvenc"),
    ("libx264",      "-c:v libx264"),
]

def has_encoder(name: str) -> bool:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"],
            text=True, stderr=subprocess.STDOUT
        )
        return re.search(rf"\b{name}\b", out) is not None
    except Exception:
        return False

def sanity_probe(flag: str, silent: bool) -> bool:
    # 1s null encode to verify the encoder opens.
    cmd = f"ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=1280x720:rate=30 -t 1 {flag} -f null -"
    try:
        # Capture ALL output so failures don't print to console
        subprocess.run(shlex.split(cmd), check=True, capture_output=silent, text=True)
        return True
    except subprocess.CalledProcessError:
        return False

def pick(preferred_csv: str|None, skip_hw: bool, silent: bool) -> str:
    candidates = BASE_CANDIDATES[:]
    if skip_hw:
        # Drop hardware encoders
        candidates = [c for c in candidates if c[0] not in ("h264_v4l2m2m", "h264_nvenc")]

    if preferred_csv:
        want = [x.strip() for x in preferred_csv.split(",") if x.strip()]
        # Reorder by user preference while keeping only available base candidates
        order = []
        base = dict(candidates)
        for n in want:
            if n in base:
                order.append((n, base[n]))
        for n, f in candidates:
            if n not in [x[0] for x in order]:
                order.append((n, f))
        candidates = order

    for name, flag in candidates:
        if has_encoder(name) and sanity_probe(flag, silent=silent):
            return flag

    # Last-ditch: try software x264 even if not listed
    if sanity_probe("-c:v libx264", silent=silent):
        return "-c:v libx264"

    raise SystemExit("No working H.264 encoder found.")

def main():
    ap = argparse.ArgumentParser(description="Pick a working H.264 encoder.")
    ap.add_argument("--preferred", help="Comma-separated encoders by preference", default=None)
    ap.add_argument("--skip-hw", action="store_true", help="Skip hardware encoders")
    ap.add_argument("--silent", action="store_true", help="Silence probe errors")
    args = ap.parse_args()
    print(pick(args.preferred, args.skip_hw, args.silent))

if __name__ == "__main__":
    main()
