from tools.audio_devices import diag

if __name__ == "__main__":
    info = diag()
    print("Pulse sources:")
    for s in info["pulse"]:
        print(f"  - {s}")
    print("\nALSA devices:")
    for d in info["alsa"]:
        print(f"  - {d}")
