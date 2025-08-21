#!/usr/bin/env python3
"""List PulseAudio sources and ALSA capture devices."""
from tools.audio_devices import list_pulse_sources, list_alsa_devices


def main() -> None:
    print("Pulse sources:")
    for s in list_pulse_sources():
        print("  -", s["name"])
    print("\nALSA devices:")
    for d in list_alsa_devices():
        print(f"  - {d['hint']} (card {d['card']} device {d['device']})")


if __name__ == "__main__":
    main()
