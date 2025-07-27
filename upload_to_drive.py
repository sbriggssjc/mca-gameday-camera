from __future__ import annotations

"""Utility to upload video files to Google Drive using OAuth2.

This module provides ``upload_to_drive`` for sending MP4 recordings to
Google Drive. It uses :mod:`PyDrive` for authentication. On systems
without a graphical environment (for example Jetson devices running
headless) the authorization flow falls back to a command line prompt.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

try:  # pragma: no cover - optional import
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
except Exception as exc:  # pragma: no cover - dependency missing
    raise ImportError(
        "PyDrive is not installed. Run `pip install PyDrive "
        "google-api-python-client oauth2client` to enable Google Drive "
        "uploads."
    ) from exc

TOKEN_FILE = "drive_token.json"


def get_drive() -> GoogleDrive:
    """Return an authenticated ``GoogleDrive`` instance.

    The function stores credentials in :data:`TOKEN_FILE`. If running on a
    system without a graphical display (for example a Jetson used headless),
    ``CommandLineAuth`` is used instead of ``LocalWebserverAuth``.
    """

    gauth = GoogleAuth()
    gauth.LoadCredentialsFile(TOKEN_FILE)
    if gauth.credentials is None:
        # Detect headless environment
        if os.getenv("DISPLAY"):
            try:
                gauth.LocalWebserverAuth()
            except Exception:
                gauth.CommandLineAuth()
        else:
            gauth.CommandLineAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    gauth.SaveCredentialsFile(TOKEN_FILE)
    return GoogleDrive(gauth)


def upload_to_drive(
    file_path: str | Path,
    folder_id: str | None = None,
    *,
    title: str | None = None,
) -> Tuple[str, str]:
    """Upload ``file_path`` to Google Drive.

    Parameters
    ----------
    file_path:
        Path to the local file to upload.
    folder_id:
        Optional destination folder ID. If ``None`` the file is placed in the
        Drive root.
    title:
        Optional name for the uploaded file. Defaults to the local file name.
    Returns
    -------
    Tuple[str, str]
        The Drive file ID and the shareable view URL.
    """

    drive = get_drive()
    path = Path(file_path)
    meta = {"title": title or path.name}
    if folder_id:
        meta["parents"] = [{"id": folder_id}]
    gfile = drive.CreateFile(meta)
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

    for f in args.files:
        path = Path(f)
        try:
            file_id, view_url = upload_to_drive(path, args.folder_id)
            if args.move:
                uploaded_dir.mkdir(parents=True, exist_ok=True)
                dest = uploaded_dir / path.name
                path.rename(dest)
        except Exception as exc:  # pragma: no cover - network/auth
            print(f"Error uploading {path}: {exc}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
