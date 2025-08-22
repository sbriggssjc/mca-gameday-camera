import json
import subprocess


def list_pulse_sources():
    try:
        out = subprocess.run(
            ["pactl", "list", "short", "sources"],
            check=True, capture_output=True, text=True
        ).stdout.strip().splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    sources = [line.split("\t")[1] for line in out]
    return sources


def list_alsa_devices():
    # Parse `arecord -l` into hw:card,device list
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    hw = []
    for line in out.splitlines():
        # card 2: ..., device 0:
        if "card " in line and "device " in line:
            parts = line.replace(":", "").split()
            c = parts[parts.index("card")+1]
            d = parts[parts.index("device")+1]
            hw.append(f"hw:{c},{d}")
    return hw


def choose_best_pulse_source(sources):
    # Prefer real mic; avoid *.monitor unless nothing else.
    preferred = [s for s in sources if ".monitor" not in s]
    return preferred[0] if preferred else (sources[0] if sources else None)


def diag():
    return {
        "pulse": list_pulse_sources(),
        "alsa": list_alsa_devices()
    }


if __name__ == "__main__":
    info = diag()
    print("Pulse sources:")
    for s in info["pulse"]:
        print(f"  - {s}")
    print("\nALSA devices:")
    for d in info["alsa"]:
        print(f"  - {d}")
