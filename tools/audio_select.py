import json, os, re, subprocess, sys

def _load_sources():
    try:
        out = subprocess.check_output(["pactl", "-f", "json", "list", "sources"], text=True)
        return json.loads(out)
    except Exception as e:
        print(f"[audio_select] ERROR: cannot list Pulse sources: {e}", file=sys.stderr)
        return []

def _score_source(src):
    name = src.get("name","").lower()
    desc = src.get("description","").lower()
    score = 0
    if any(k in desc for k in ["video", "mic", "rode"]) or "usb" in desc:
        score += 50
    if "mono" in name or "mono" in desc:
        score += 10
    if "monitor" in name or "monitor" in desc:
        score -= 100
    if "hdmi" in name or "hdmi" in desc:
        score -= 50
    if "analog-stereo" in name or "analog-stereo" in desc:
        score += 2
    return score

def _match_regex(sources, pattern):
    rx = re.compile(pattern, re.I)
    for s in sources:
        if rx.search(s.get("name","")) or rx.search(s.get("description","")):
            return s
    return None

def pick_pulse_source():
    sources = _load_sources()
    if not sources:
        return None

    exact = os.environ.get("PULSE_DEV")
    if exact:
        for s in sources:
            if s.get("name") == exact:
                return s

    hint = os.environ.get("PULSE_DEV_REGEX")
    if hint:
        m = _match_regex(sources, hint)
        if m:
            return m

    for pat in [r"VideoMic", r"R.?DE", r"usb.*mic", r"mic.*usb"]:
        m = _match_regex(sources, pat)
        if m:
            return m

    ranked = sorted(
        [s for s in sources if "monitor" not in s.get("name","").lower()],
        key=_score_source,
        reverse=True,
    )
    return ranked[0] if ranked else None

if __name__ == "__main__":
    sel = pick_pulse_source()
    print(sel.get("name") if sel else "", end="")
