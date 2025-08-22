#!/usr/bin/env python3
"""
Select the best PulseAudio source for streaming.
Preference order:
  1) Rode VideoMic GO II (usb, not monitor)
  2) Any usb mic (alsa_input.*)
  3) Fallback to default source
Outputs the Pulse source name on stdout.
"""
import re, subprocess, sys

PREFERRED_PATTERNS = [
    r"usb.*VideoMic.*GO.*II.*(mono|analog).*",   # Rode
    r"alsa_input\..*",                           # any input (not monitor)
]
EXCLUDE_PATTERNS = [r"\.monitor$"]

def list_pulse_sources():
    out = subprocess.check_output(["pactl", "list", "short", "sources"], text=True)
    # columns: index\tname\tdriver\tstate\t...
    names = [line.split("\t")[1] for line in out.strip().splitlines() if line.strip()]
    return names

def pick():
    names = list_pulse_sources()
    for pat in PREFERRED_PATTERNS:
        cre = re.compile(pat, re.I)
        for n in names:
            if any(re.search(x, n) for x in EXCLUDE_PATTERNS):
                continue
            if cre.search(n):
                print(n)
                return 0
    # fallback: first non-monitor
    for n in names:
        if not any(re.search(x, n) for x in EXCLUDE_PATTERNS):
            print(n)
            return 0
    # last resort
    print(names[0] if names else "", end="")
    return 0

if __name__ == "__main__":
    sys.exit(pick())
