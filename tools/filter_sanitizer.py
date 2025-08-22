#!/usr/bin/env python3
import re

# Replace any aresample chain that includes the risky min/max comp flags with a safe variant
SAFE = "aresample=async=1:first_pts=0"
MIN = "min" + "_comp"
MAX = "max" + "_comp"

def sanitize_aresample(s: str) -> str:
    if not s:
        return SAFE
    # collapse whitespace/newlines to simplify matching
    flat = re.sub(r"\s+", " ", s)
    # remove any min or max comp fields if present
    flat = re.sub(r",\s*" + MIN + r"\s*=\s*[^,:\)\s]+", "", flat)
    flat = re.sub(r",\s*" + MAX + r"\s*=\s*[^,:\)\s]+", "", flat)
    # if someone built aresample=async=1:first_pts=0 with other args, leave them
    # but if we still see either flag, hard replace the whole aresample with SAFE
    if re.search(r"aresample\s*=\s*[^\"']*(" + MIN + "|" + MAX + ")", flat):
        flat = re.sub(r"aresample\s*=\s*[^\"']*", SAFE, flat)
    # ensure at least one aresample exists; if none, append SAFE at end
    if "aresample=" not in flat:
        flat = (flat.rstrip(", ") + ("," if flat.strip() else "")) + SAFE
    return flat

if __name__ == "__main__":
    import sys,json
    text = sys.stdin.read()
    print(sanitize_aresample(text))
