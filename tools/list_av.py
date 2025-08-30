#!/usr/bin/env python3
import glob
import subprocess


def _run(cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return ""


print("Video devices:")
videos = sorted(glob.glob("/dev/video*"))
if videos:
    for v in videos:
        print(f"  - {v}")
else:
    print("  (none)")

print("\nPulse sources:")
ps = _run(["pactl", "list", "short", "sources"])
if ps:
    for line in ps.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            print(f"  - {parts[1]}")
else:
    print("  (none)")
