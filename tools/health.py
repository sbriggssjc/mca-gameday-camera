#!/usr/bin/env python3
"""Simple health monitor for gameday logs."""
from pathlib import Path
import re, sys, time

LOG_DIR = Path("logs")
PATTERNS = [
    (re.compile(r"Connection to tcp://.* failed"), "ingest_failure"),
    (re.compile(r"No such device"), "device_missing"),
]

def latest_log():
    files = sorted(LOG_DIR.glob("ffmpeg_*.log"))
    return files[-1] if files else None

def monitor(path: Path) -> int:
    with path.open() as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            for cre, tag in PATTERNS:
                if cre.search(line):
                    print(f"[health] {tag}: {line.strip()}")
    return 0

def main() -> int:
    log = latest_log()
    if not log:
        print("[health] no log files found")
        return 1
    print(f"[health] monitoring {log}")
    return monitor(log)

if __name__ == "__main__":
    sys.exit(main())
