import os
import argparse
import mimetypes

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
TOKEN_FILE = "token.json"
CLIENT_SECRETS = "client_secrets.json"


def get_authenticated_service():
    """Authenticate and return a YouTube service resource."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                print(f"Failed to refresh credentials: {exc}")
                creds = None
        if not creds:
            if not os.path.exists(CLIENT_SECRETS):
                raise FileNotFoundError(
                    f"{CLIENT_SECRETS} not found. See README for setup instructions." )
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, SCOPES)
            creds = flow.run_console()
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def upload_video(service, file_path: str, title: str, description: str, privacy: str):
    """Upload a video to YouTube and return the video id."""
    mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
    request = service.videos().insert(
        part="snippet,status",
        body={
            "snippet": {"title": title, "description": description},
            "status": {"privacyStatus": privacy},
        },
        media_body=media,
    )
    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"Upload progress: {int(status.progress() * 100)}%")
        except HttpError as err:
            if err.resp.status in {403, 429}:
                raise RuntimeError("Upload failed: quota exceeded") from err
            raise
    return response.get("id")


def main():
    parser = argparse.ArgumentParser(description="Upload a video to YouTube")
    parser.add_argument("--file", required=True, help="Path to the video file")
    parser.add_argument("--title", required=True, help="Video title")
    parser.add_argument("--description", default="", help="Video description")
    parser.add_argument(
        "--privacy",
        choices=["public", "unlisted", "private"],
        default="unlisted",
        help="Video privacy status",
    )
    args = parser.parse_args()

    try:
        service = get_authenticated_service()
        video_id = upload_video(service, args.file, args.title, args.description, args.privacy)
        print(f"Video uploaded: https://youtu.be/{video_id}")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
