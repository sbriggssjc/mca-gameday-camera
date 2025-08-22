"""Upload completed game video to Google Drive."""

from __future__ import annotations

import os
from pathlib import Path

from upload_to_drive import upload_to_drive


def upload_game(path: str, folder_id: str | None = None) -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    if folder_id is None:
        folder_id = os.getenv("GDRIVE_FOLDER_ID")
    if not folder_id:
        raise RuntimeError("GDRIVE_FOLDER_ID not set")
    upload_to_drive(str(file_path), folder_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload game recording to Google Drive")
    parser.add_argument("file", help="path to recording")
    parser.add_argument("--folder-id", default=os.getenv("GDRIVE_FOLDER_ID"), help="Google Drive folder ID")
    args = parser.parse_args()
    try:
        upload_game(args.file, args.folder_id)
        print("Upload successful")
    except Exception as exc:
        print(f"Upload failed: {exc}")
