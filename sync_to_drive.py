"""Upload MP4 files from the video folder to Google Drive."""

from __future__ import annotations

import argparse
from pathlib import Path


def upload_files(folder_id: str, files: list[Path]) -> None:
    """Upload the given files to the provided Drive folder."""
    try:
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
    except Exception as exc:  # pragma: no cover - optional
        raise ImportError("PyDrive is required for uploading") from exc

    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    for file_path in files:
        gfile = drive.CreateFile({"title": file_path.name, "parents": [{"id": folder_id}]})
        gfile.SetContentFile(str(file_path))
        gfile.Upload()
        print(f"Uploaded {file_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync MP4 recordings to Google Drive")
    parser.add_argument("--folder-id", required=True, help="Drive folder ID to upload to")
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading and just list files")
    args = parser.parse_args()

    mp4_files = sorted(Path("video").glob("*.mp4"))
    if not mp4_files:
        print("No MP4 files found in video/")
        return

    if args.no_upload:
        for path in mp4_files:
            print(f"Found {path.name}")
        return

    upload_files(args.folder_id, mp4_files)


if __name__ == "__main__":  # pragma: no cover - script
    main()
