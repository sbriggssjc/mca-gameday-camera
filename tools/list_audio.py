# tools/list_audio.py
import sys
from pathlib import Path

# Make repo root importable whether run as "python3 tools/list_audio.py" or "-m tools.list_audio"
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from tools.audio_devices import list_pulse_sources, list_alsa_devices
except Exception as e:
    print(f"[list_audio] Import error: {e}")
    sys.exit(1)

def main():
    print("Pulse sources:")
    for s in list_pulse_sources():
        print("  -", s["name"])
    print("\nALSA devices:")
    for d in list_alsa_devices():
        print(f"  - {d['hint']} (card {d['card']} device {d['device']})")

if __name__ == "__main__":
    main()
