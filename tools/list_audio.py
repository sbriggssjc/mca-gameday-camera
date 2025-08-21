"""List available audio devices.

Usage:
  PYTHONPATH=. python3 -m tools.list_audio
  PYTHONPATH=. python3 -m tools.list_audio --json
"""

import argparse, json

from tools.audio_devices import diag


def main() -> None:
    parser = argparse.ArgumentParser(description="List audio devices")
    parser.add_argument("--json", action="store_true", help="output JSON instead of text")
    args = parser.parse_args()

    info = diag()
    if args.json:
        print(json.dumps(info))
        return

    print("Pulse sources:")
    for s in info["pulse"]:
        print(f"  - {s}")
    print("\nALSA devices:")
    for d in info["alsa"]:
        print(f"  - {d}")


if __name__ == "__main__":
    main()
