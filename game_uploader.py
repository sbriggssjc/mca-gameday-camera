"""Upload completed game video to Google Drive."""

from __future__ import annotations

import subprocess
from pathlib import Path


def upload_game(path: str, dest: str = "gdrive:/MCA/GameDayFull/") -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    result = subprocess.run(["rclone", "copy", str(file_path), dest], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload game recording to Google Drive")
    parser.add_argument("file", help="path to recording")
    parser.add_argument("--dest", default="gdrive:/MCA/GameDayFull/", help="rclone destination")
    args = parser.parse_args()
    try:
        upload_game(args.file, args.dest)
        print("Upload successful")
    except Exception as exc:
        print(f"Upload failed: {exc}")
