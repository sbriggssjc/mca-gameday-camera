from __future__ import annotations

"""Utility to upload files to Google Drive with clear logging."""

import os
from pathlib import Path

from gdrive_utils import upload_to_google_drive


def maybe_upload(file_path: str) -> None:
    drive_sync_enabled = os.getenv("GOOGLE_DRIVE_SYNC") == "1"
    if not drive_sync_enabled:
        print("[gdrive] sync disabled (set GOOGLE_DRIVE_SYNC=1)", flush=True)
        return
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds or not Path(creds).exists():
        print("[gdrive] missing GOOGLE_APPLICATION_CREDENTIALS", flush=True)
        return
    folder = os.getenv("GDRIVE_UPLOAD_FOLDER") or os.getenv("GDRIVE_FOLDER_ANALYZED")
    if not folder:
        print("[gdrive] missing target folder", flush=True)
        return
    print(f"[gdrive] target folder {folder}", flush=True)
    print("[gdrive] starting upload", flush=True)
    upload_to_google_drive(file_path, folder)
    print("[gdrive] upload done", flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m sync.gdrive_uploader <file>")
        raise SystemExit(1)
    maybe_upload(sys.argv[1])

