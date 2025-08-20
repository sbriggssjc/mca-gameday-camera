import subprocess, json, re


def list_pulse_sources():
    try:
        out = subprocess.check_output(["pactl", "list", "sources", "short"], text=True)
        rows = []
        for line in out.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                rows.append({"index": parts[0], "name": parts[1]})
        return rows
    except Exception:
        return []


def default_pulse_source():
    candidates = [r for r in list_pulse_sources() if "monitor" not in r["name"]]
    return candidates[0]["name"] if candidates else None


def list_alsa_devices():
    try:
        out = subprocess.check_output(["arecord", "-l"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return []
    cards = []
    for line in out.splitlines():
        m = re.search(r"card (\d+): ([^,]+), device (\d+): ([^\[]+)", line)
        if m:
            card, cardname, dev, devname = m.groups()
            cards.append({"card": card, "device": dev, "hint": f"hw:{card},{dev}"})
    return cards


def default_alsa_device():
    devs = list_alsa_devices()
    return devs[0]["hint"] if devs else None


def pick_audio_source(backend, pulse_name, alsa_dev):
    if backend in (None, "", "auto"):
        pulse = pulse_name or default_pulse_source()
        if pulse:
            return ("pulse", pulse)
        alsa = alsa_dev or default_alsa_device()
        if alsa:
            return ("alsa", alsa)
        return (None, None)
    if backend == "pulse":
        return ("pulse", pulse_name or default_pulse_source())
    if backend == "alsa":
        return ("alsa", alsa_dev or default_alsa_device())
    return (None, None)
