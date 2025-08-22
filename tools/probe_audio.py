#!/usr/bin/env python3
"""
Probe a Pulse source with ffmpeg volumedetect for ~2s.
Print JSON: {"rc": int, "mean_db": float, "peak_db": float}
Exit nonzero if capture fails.
"""
import json, re, subprocess, sys

SOURCE = sys.argv[1] if len(sys.argv) > 1 else ""
if not SOURCE:
    print(json.dumps({"rc": 2, "error": "no_source"}))
    sys.exit(2)

cmd = [
  "ffmpeg","-hide_banner","-nostats","-f","pulse","-i",SOURCE,
  "-t","2","-af","volumedetect","-f","null","/dev/null"
]
p = subprocess.run(cmd, capture_output=True, text=True)
stdout = (p.stdout or "") + (p.stderr or "")
mm = re.search(r"mean_volume:\s*([-+]?\d+(?:\.\d+)?)\s*dB", stdout)
pk = re.search(r"max_volume:\s*([-+]?\d+(?:\.\d+)?)\s*dB", stdout)
mean_db = float(mm.group(1)) if mm else None
peak_db = float(pk.group(1)) if pk else None
print(json.dumps({"rc": p.returncode, "mean_db": mean_db, "peak_db": peak_db}))
sys.exit(p.returncode)
