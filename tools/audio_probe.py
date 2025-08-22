import json, re, subprocess

VOL_RE = re.compile(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB.*?max_volume:\s*(-?\d+\.?\d*)\s*dB", re.S)


def probe_pulse_source(src, seconds=2):
    # Use null muxer; capture stderr (ffmpeg prints volumedetect to stderr)
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-f", "pulse", "-i", src,
        "-t", str(seconds),
        "-af", "volumedetect",
        "-f", "null", "/dev/null"
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        stderr = p.stderr or ""
    except FileNotFoundError:
        return {"rc": 1, "mean_db": None, "peak_db": None, "log": "ffmpeg not found"}
    m = VOL_RE.search(stderr)
    mean = float(m.group(1)) if m else None
    peak = float(m.group(2)) if m else None
    return {
        "rc": p.returncode,
        "mean_db": mean,
        "peak_db": peak,
        "log": stderr[-2000:]  # tail for logs
    }


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "default"
    print(json.dumps(probe_pulse_source(src)))
