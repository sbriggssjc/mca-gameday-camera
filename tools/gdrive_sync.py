from __future__ import annotations

import os
import io
import json
import mimetypes
import hashlib
import datetime
from pathlib import Path
from typing import Optional, List, Dict

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# Only function and constant definitions here; no code executed at import time.

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _drive():
    creds_json = os.getenv("GDRIVE_CREDENTIALS_JSON")
    if not creds_json or not Path(creds_json).exists():
        raise RuntimeError("GDRIVE_CREDENTIALS_JSON not set or file missing.")
    creds = service_account.Credentials.from_service_account_file(
        creds_json, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _build_folder_query(name: str, parent_id: Optional[str] = None) -> str:
    esc = name.replace("'", "\\'")
    base = (
        "mimeType='application/vnd.google-apps.folder' and "
        "name='{name}' and trashed=false"
    ).format(name=esc)
    if parent_id:
        base += " and '{pid}' in parents".format(pid=parent_id)
    return base


def find_or_create_folder(drive, name: str, parent_id: Optional[str] = None) -> str:
    """Return id of existing Drive folder or create one."""
    q = _build_folder_query(name, parent_id)
    resp = drive.files().list(q=q, fields="files(id,name)").execute()
    files = resp.get("files", [])
    if files:
        return files[0]["id"]
    body = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        body["parents"] = [parent_id]
    f = drive.files().create(body=body, fields="id").execute()
    return f["id"]


# Backwards compatibility alias
def ensure_folder(drive, name: str, parent_id: Optional[str] = None) -> str:
    return find_or_create_folder(drive, name, parent_id)


def get_or_create_subpath(
    drive, parent_id: str, rel_path: Path, cache: Optional[Dict[str, str]] = None
) -> str:
    cache = cache if cache is not None else {}
    cur = parent_id
    parts = [p for p in rel_path.parts if p not in ("", ".")]
    acc: List[str] = []
    for part in parts:
        acc.append(part)
        key = "/".join(acc)
        if key in cache:
            cur = cache[key]
            continue
        cur = find_or_create_folder(drive, part, cur)
        cache[key] = cur
    return cur


def list_files(
    parent_id: str,
    query: Optional[str] = None,
    query_since_iso: Optional[str] = None,
) -> List[Dict[str, str]]:
    drive = _drive()
    q = f"'{parent_id}' in parents and trashed=false"
    if query:
        q += f" and {query}"
    if query_since_iso:
        q += f" and modifiedTime > '{query_since_iso}'"
    files: List[Dict[str, str]] = []
    page_token = None
    while True:
        resp = (
            drive.files()
            .list(
                q=q,
                fields="nextPageToken, files(id,name,size,md5Checksum,modifiedTime)",
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def download_file(file_id: str, dest_path: Path) -> None:
    drive = _drive()
    request = drive.files().get_media(fileId=file_id)
    fh = io.FileIO(dest_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()


def hash_file(path: Path, algo: str = "md5") -> str:
    h = hashlib.md5() if algo.lower() == "md5" else hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_file_with_md5(drive, local_path: Path, parent_id: str) -> Dict[str, str]:
    mime, _ = mimetypes.guess_type(str(local_path))
    chunk_mb = int(os.getenv("UPLOAD_CHUNK_MB", "8"))
    media = MediaFileUpload(
        str(local_path),
        mimetype=mime,
        resumable=True,
        chunksize=chunk_mb * 1024 * 1024,
    )
    body = {"name": local_path.name, "parents": [parent_id]}
    req = drive.files().create(body=body, media_body=media, fields="id, md5Checksum")
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
    return resp


def upload_tree_with_manifest(local_dir: Path, parent_id: str, manifest_path: Path) -> None:
    drive = _drive()
    cache: Dict[str, str] = {}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf8") as mf:
        for root, dirs, files in os.walk(local_dir):
            root_p = Path(root)
            rel_root = root_p.relative_to(local_dir)
            drive_parent = get_or_create_subpath(drive, parent_id, rel_root, cache)
            for fname in files:
                lp = root_p / fname
                md5 = hash_file(lp, "md5")
                sha1 = hash_file(lp, "sha1")
                up = upload_file_with_md5(drive, lp, drive_parent)
                file_id = up["id"]
                meta = (
                    drive.files()
                    .get(fileId=file_id, fields="id,name,md5Checksum,size,parents")
                    .execute()
                )
                drive_md5 = meta.get("md5Checksum")
                status = "verified" if drive_md5 and drive_md5 == md5 else "unverified"
                rec = {
                    "ts": datetime.datetime.utcnow().isoformat(),
                    "local": str(lp),
                    "bytes": lp.stat().st_size,
                    "md5": md5,
                    "sha1": sha1,
                    "drive_id": file_id,
                    "drive_md5": drive_md5,
                    "parents": meta.get("parents", []),
                    "status": status,
                }
                mf.write(json.dumps(rec) + "\n")


# Backwards compatibility exports
__all__ = [
    "find_or_create_folder",
    "ensure_folder",
    "get_or_create_subpath",
    "list_files",
    "download_file",
    "hash_file",
    "upload_file_with_md5",
    "upload_tree_with_manifest",
]

