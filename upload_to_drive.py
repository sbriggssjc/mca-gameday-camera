from __future__ import annotations
import subprocess
from pathlib import Path
import argparse
import os


def upload_file(file_path: str | Path, folder_id: str) -> None:
    """Upload a single file to Google Drive using the gdrive CLI."""
    file_str = str(file_path)
    result = subprocess.run([
        "gdrive",
        "upload",
        "--parent",
        folder_id,
        file_str,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    print(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload files to Google Drive via gdrive CLI")
    parser.add_argument("files", nargs="+", help="MP4 files to upload")
    parser.add_argument(
        "--folder-id",
        default=os.getenv("GDRIVE_FOLDER_ID"),
        help="Destination Google Drive folder ID (or set GDRIVE_FOLDER_ID)",
    )
    args = parser.parse_args()
    if not args.folder_id:
        parser.error("--folder-id is required (or set GDRIVE_FOLDER_ID)")
    for f in args.files:
        upload_file(f, args.folder_id)


if __name__ == "__main__":
    main()
