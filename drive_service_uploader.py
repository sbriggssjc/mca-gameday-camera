"""Upload new video files from the local ``video/`` folder to Google Drive.

This script uses a service account for authentication. Any ``.mp4`` files in the
``video/`` directory that are not already present in the destination Drive folder
are uploaded. A log entry is written for each attempted upload.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def get_service(creds_path: str):
    """Return an authorized Drive service using the given credentials file."""
    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def remote_file_exists(service, folder_id: str, name: str) -> bool:
    """Return True if a file with ``name`` exists in ``folder_id``."""
    query = (
        f"name='{name}' and '{folder_id}' in parents and trashed=false"
    )
    resp = service.files().list(q=query, fields="files(id)").execute()
    return bool(resp.get("files"))


def upload_file(service, folder_id: str, path: Path) -> bool:
    """Upload ``path`` to the Drive ``folder_id``. Return True on success."""
    try:
        metadata = {"name": path.name, "parents": [folder_id]}
        media = MediaFileUpload(path, resumable=True)
        service.files().create(body=metadata, media_body=media).execute()
        return True
    except Exception as exc:  # pragma: no cover - network
        logging.error("Failed to upload %s: %s", path.name, exc)
        return False


def sync_videos(service, folder_id: str, files: Iterable[Path]) -> None:
    """Upload each file if it does not already exist remotely."""
    for path in files:
        if remote_file_exists(service, folder_id, path.name):
            logging.info("Skipping existing file %s", path.name)
            continue
        success = upload_file(service, folder_id, path)
        if success:
            logging.info("Uploaded %s", path.name)
        else:
            logging.error("Upload failed for %s", path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync videos to Google Drive")
    parser.add_argument("--credentials", required=True, help="Path to service account JSON")
    parser.add_argument("--folder-id", required=True, help="Destination Drive folder ID")
    parser.add_argument(
        "--log", default="drive_upload.log", help="Path to log file"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log, format="%(asctime)s %(levelname)s %(message)s")

    service = get_service(args.credentials)

    video_files = sorted(Path("video").glob("*.mp4"))
    if not video_files:
        print("No video files found to upload")
        return

    sync_videos(service, args.folder_id, video_files)

    print("Upload complete")


if __name__ == "__main__":  # pragma: no cover - script
    main()
