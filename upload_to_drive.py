from __future__ import annotations

"""Utility to upload video files to Google Drive using OAuth2."""

import argparse
import os
from pathlib import Path
from typing import Tuple

try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
except Exception as exc:  # pragma: no cover - optional import
    raise ImportError(
        "PyDrive is required for Google Drive uploads"
    ) from exc

TOKEN_FILE = "drive_token.json"


def get_drive() -> GoogleDrive:
    """Return an authenticated ``GoogleDrive`` instance."""
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile(TOKEN_FILE)
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    gauth.SaveCredentialsFile(TOKEN_FILE)
    return GoogleDrive(gauth)


def upload_to_drive(file_path: str | Path, folder_id: str) -> Tuple[str, str]:
    """Upload ``file_path`` to ``folder_id``. Return ``(file_id, view_url)``."""
    drive = get_drive()
    path = Path(file_path)
    gfile = drive.CreateFile({"title": path.name, "parents": [{"id": folder_id}]})
    gfile.SetContentFile(str(path))
    gfile.Upload()
    file_id = gfile["id"]
    view_url = f"https://drive.google.com/file/d/{file_id}/view"
    print(f"Uploaded {path.name} -> {view_url}")
    return file_id, view_url


def main() -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="Upload files to Google Drive")
    parser.add_argument("files", nargs="+", help="MP4 files to upload")
    parser.add_argument(
        "--folder-id",
        default=os.getenv("GDRIVE_FOLDER_ID"),
        help="Destination Google Drive folder ID (or set GDRIVE_FOLDER_ID)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move uploaded files to video/uploaded/",
    )
    args = parser.parse_args()
    if not args.folder_id:
        parser.error("--folder-id is required (or set GDRIVE_FOLDER_ID)")

    uploaded_dir = Path("video/uploaded")
    if args.move:
        uploaded_dir.mkdir(parents=True, exist_ok=True)

    drive = get_drive()
    for f in args.files:
        path = Path(f)
        try:
            gfile = drive.CreateFile({"title": path.name, "parents": [{"id": args.folder_id}]})
            gfile.SetContentFile(str(path))
            gfile.Upload()
            file_id = gfile["id"]
            view_url = f"https://drive.google.com/file/d/{file_id}/view"
            print(f"Uploaded {path.name} -> {view_url}")
            if args.move:
                dest = uploaded_dir / path.name
                path.rename(dest)
        except Exception as exc:  # pragma: no cover - network/auth
            print(f"Error uploading {path}: {exc}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
