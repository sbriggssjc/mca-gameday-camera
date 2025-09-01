import json, os, re, subprocess, sys


def _load_sources():
    try:
        out = subprocess.check_output(["pactl", "-f", "json", "list", "sources"], text=True)
        return json.loads(out)
    except Exception as e:
        print(f"[audio_select] ERROR: cannot list Pulse sources: {e}", file=sys.stderr)
        return []


def _score_source(src):
    # Favor external USB mics, mono fallback is fine, avoid HDMI/stereo monitors
    name = src.get("name", "").lower()
    desc = src.get("description", "").lower()
    score = 0
    if "video" in desc or "mic" in desc or "rode" in desc or "usb" in desc:
        score += 50
    if "mono" in desc or "mono" in name:
        score += 10
    if "monitor" in name or "monitor" in desc:
        score -= 100  # not a capture device
    if "hdmi" in name or "hdmi" in desc:
        score -= 50
    if "analog-stereo" in name or "analog-stereo" in desc:
        score += 2
    return score


def _match_regex(sources, pattern):
    rx = re.compile(pattern, re.I)
    for s in sources:
        if rx.search(s.get("name", "")) or rx.search(s.get("description", "")):
            return s
    return None


def pick_pulse_source():
    sources = _load_sources()
    if not sources:
        return None

    # 1) Exact env override (legacy): PULSE_DEV (exact name)
    exact = os.environ.get("PULSE_DEV")
    if exact:
        for s in sources:
            if s.get("name") == exact:
                return s

    # 2) Regex hint: PULSE_DEV_REGEX (e.g., "VideoMic|RØDE|R__DE")
    hint = os.environ.get("PULSE_DEV_REGEX")
    if hint:
        m = _match_regex(sources, hint)
        if m:
            return m

    # 3) Known good Rode / external patterns
    for pat in [r"VideoMic", r"R.?DE", r"usb.*mic", r"mic.*usb"]:
        m = _match_regex(sources, pat)
        if m:
            return m

    # 4) Best-scored non-monitor
    ranked = sorted(
        [s for s in sources if "monitor" not in s.get("name", "").lower()],
        key=_score_source,
        reverse=True,
    )
    return ranked[0] if ranked else None


def pulse_name_from_selection(s):
    return s.get("name") if s else None


if __name__ == "__main__":
    sel = pick_pulse_source()
    name = pulse_name_from_selection(sel)
    if not name:
        print("", end="")
        sys.exit(1)
    print(name)
