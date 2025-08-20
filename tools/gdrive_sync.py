import os
import mimetypes
from pathlib import Path
from typing import Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _drive():
    creds_json = os.getenv("GDRIVE_CREDENTIALS_JSON")
    if not creds_json or not Path(creds_json).exists():
        raise RuntimeError("GDRIVE_CREDENTIALS_JSON not set or file missing.")
    creds = service_account.Credentials.from_service_account_file(creds_json, scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def ensure_folder(drive, name: str, parent_id: Optional[str] = None) -> str:
    q = f"mimeType='application/vnd.google-apps.folder' and name='{name.replace("'", "\\'")}' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    resp = drive.files().list(q=q, fields="files(id,name)").execute()
    files = resp.get("files", [])
    if files:
        return files[0]["id"]
    body = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        body["parents"] = [parent_id]
    f = drive.files().create(body=body, fields="id").execute()
    return f["id"]


def upload_file(drive, local_path: Path, parent_id: str) -> str:
    mime, _ = mimetypes.guess_type(str(local_path))
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True, chunksize=8*1024*1024)
    body = {"name": local_path.name, "parents": [parent_id]}
    req = drive.files().create(body=body, media_body=media, fields="id, name, parents")
    resp = None
    while resp is None:
        status, resp = req.next_chunk()
    return resp["id"]


def upload_tree(local_dir: Path, parent_id: str) -> None:
    drive = _drive()
    # Build a mapping of subfolders in Drive
    folder_cache = {str(local_dir): parent_id}
    for root, dirs, files in os.walk(local_dir):
        root_p = Path(root)
        # ensure Drive folder for this root
        parent_drive_id = folder_cache[str(root_p)]
        for d in dirs:
            d_id = ensure_folder(drive, d, parent_drive_id)
            folder_cache[str(root_p / d)] = d_id
        # upload files in this folder
        for f in files:
            upload_file(drive, root_p / f, parent_drive_id)
