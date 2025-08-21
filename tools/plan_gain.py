#!/usr/bin/env python3
"""
Given mean dBFS, suggest ffmpeg filter gain so target ~ -16 LUFS-ish program level.
Simple rule:
- If mean <= -35 dB, plan +15 dB
- If -35 < mean <= -25 dB, plan +10 dB
- If -25 < mean <= -18 dB, plan +6 dB
- Else 0 dB (use loudnorm to tame peaks)
Print a filter fragment for ffmpeg (-af '...').
"""
import sys

mean = float(sys.argv[1])
gain = 0.0
if mean <= -35:
    gain = 15
elif mean <= -25:
    gain = 10
elif mean <= -18:
    gain = 6
print(f"volume={gain}dB")
