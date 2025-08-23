#!/usr/bin/env python3
import subprocess, json, sys


def _run(cmd):
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except Exception:
        out = ""
    return out


print("Pulse sources:")
ps = _run(["pactl", "list", "short", "sources"])
if ps:
    for line in ps.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            print(f"  - {parts[1]}")
else:
    print("  (none)")

print("\nALSA devices:")
arec = _run(["arecord", "-l"])
for line in arec.splitlines():
    line = line.strip()
    if line.startswith("card "):
        # best-effort extract hw:X,Y
        import re
        m = re.search(r'card (\d+): .*? device (\d+):', line)
        if m:
            print(f"  - hw:{m.group(1)},{m.group(2)}")
