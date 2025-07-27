#!/usr/bin/env python3
"""Print available H.264 encoders from ffmpeg."""
import subprocess

def main() -> None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"], capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        print("ffmpeg not found")
        return
    for line in result.stdout.splitlines():
        if "264" in line:
            print(line)

if __name__ == "__main__":
    main()
