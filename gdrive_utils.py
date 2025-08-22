"""Google Drive upload helpers using PyDrive."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
except Exception as exc:  # pragma: no cover - optional import
    raise ImportError(
        "PyDrive is required for Google Drive uploads"
    ) from exc

TOKEN_FILE = "drive_token.json"


def _get_drive() -> GoogleDrive:
    """Authenticate and return a ``GoogleDrive`` instance."""
    gauth = GoogleAuth()

    # Attempt to load existing credentials
    try:
        gauth.LoadCredentialsFile(TOKEN_FILE)
    except Exception:
        print("[\u26A0\uFE0F] No drive_token.json found. Starting OAuth flow.")

    if gauth.credentials is None:
        if not os.path.exists("client_secrets.json"):
            print("[\u274C] Missing client_secrets.json. Cannot authenticate with Google Drive.")
            print("Visit https://console.cloud.google.com to create and download it.")
            raise RuntimeError("client_secrets.json missing")

        try:
            if os.getenv("DISPLAY"):
                gauth.LocalWebserverAuth()
            else:
                gauth.CommandLineAuth()
            gauth.SaveCredentialsFile(TOKEN_FILE)
            print("[\u2705] OAuth success. Credentials saved to drive_token.json")
        except Exception as exc:
            raise RuntimeError(f"Google Drive login failed: {exc}") from exc
    else:
        if gauth.access_token_expired:
            gauth.Refresh()
        gauth.Authorize()

    gauth.SaveCredentialsFile(TOKEN_FILE)
    return GoogleDrive(gauth)


def _ensure_folder(drive: GoogleDrive, name: str) -> str:
    """Return folder ID for ``name``, creating it if necessary."""
    query = (
        f"title='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = drive.ListFile({"q": query}).GetList()
    if results:
        return results[0]["id"]
    meta = {"title": name, "mimeType": "application/vnd.google-apps.folder"}
    folder = drive.CreateFile(meta)
    folder.Upload()
    return folder["id"]


def upload_to_google_drive(local_path: str | Path, drive_folder: str | None = None) -> bool:
    """Upload ``local_path`` to Google Drive.

    Parameters
    ----------
    local_path:
        File path to upload.
    drive_folder:
        Optional Drive folder name to place the file in. The folder is created if
        it does not exist.

    Returns
    -------
    bool
        ``True`` on success, ``False`` otherwise.
    """
    try:
        drive = _get_drive()
        path = Path(local_path)
        meta = {"title": path.name}
        if drive_folder:
            folder_id = _ensure_folder(drive, drive_folder)
            meta["parents"] = [{"id": folder_id}]
        gfile = drive.CreateFile(meta)
        gfile.SetContentFile(str(path))
        gfile.Upload()
        view_url = f"https://drive.google.com/file/d/{gfile['id']}/view"
        print(f"Uploaded {path.name} -> {view_url}")
        return True
    except Exception as exc:  # pragma: no cover - network/auth
        print(f"Failed to upload {local_path}: {exc}")
        return False

__all__ = ["upload_to_google_drive"]
