import argparse
import os
from datetime import datetime, timezone

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
CLIENT_SECRETS_FILE = "client_secret.json"
TOKEN_FILE = "token.json"


def authenticate_youtube(reset_auth: bool = False):
    """Authenticate user and return a YouTube service resource.

    Parameters
    ----------
    reset_auth:
        If True, delete any cached token and trigger a new OAuth flow.
    """
    if reset_auth and os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)

    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                print(f"Failed to refresh credentials: {exc}. Starting new auth flow.")
                creds = None
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES
            )
            creds = flow.run_console()
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def create_stream(service, title: str):
    """Create a liveStream and return its id and ingestion info."""
    request = service.liveStreams().insert(
        part="snippet,cdn",
        body={
            "snippet": {"title": title},
            "cdn": {
                "frameRate": "30fps",
                "ingestionType": "rtmp",
                "resolution": "720p",
            },
        },
    )
    stream = request.execute()
    stream_id = stream.get("id")
    ingestion_info = stream.get("cdn", {}).get("ingestionInfo", {})
    if not (stream_id and ingestion_info.get("ingestionAddress") and ingestion_info.get("streamName")):
        raise RuntimeError("Could not retrieve the stream URL or stream key.")
    print(f"Created stream: {stream_id}")
    return stream_id, ingestion_info


def create_broadcast(service, title: str, stream_id: str, start_time: str):
    """Create a liveBroadcast, bind it to the stream and return broadcast id."""
    body = {
        "snippet": {"title": title, "scheduledStartTime": start_time},
        "status": {"privacyStatus": "public"},
    }
    broadcast = (
        service.liveBroadcasts()
        .insert(part="snippet,contentDetails,status", body=body)
        .execute()
    )
    broadcast_id = broadcast.get("id")
    if not broadcast_id:
        raise RuntimeError("Failed to create broadcast")
    service.liveBroadcasts().bind(
        part="id,contentDetails", id=broadcast_id, streamId=stream_id
    ).execute()
    print(f"Created broadcast: {broadcast_id}")
    return broadcast_id


def get_stream_info(service, stream_id: str):
    """Fetch stream ingestion info for debugging."""
    try:
        response = (
            service.liveStreams()
            .list(part="cdn", id=stream_id)
            .execute()
        )
    except HttpError as err:
        raise RuntimeError(f"Failed to retrieve stream info: {err}") from err

    try:
        item = response["items"][0]
        ingestion = item["cdn"]["ingestionInfo"]
    except (IndexError, KeyError):
        raise RuntimeError("Could not retrieve the stream URL or stream key.")
    return ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Create YouTube livestream")
    parser.add_argument("title", help="Broadcast title")
    parser.add_argument(
        "--start-time",
        default=datetime.now(timezone.utc).isoformat(),
        help="RFC3339 start time",
    )
    parser.add_argument(
        "--reset-auth", action="store_true", help="Force OAuth re-authentication"
    )
    args = parser.parse_args()

    service = authenticate_youtube(reset_auth=args.reset_auth)

    try:
        stream_id, ingestion = create_stream(service, args.title)
        broadcast_id = create_broadcast(
            service, args.title, stream_id, args.start_time
        )
        print("Broadcast ID:", broadcast_id)
        print("Stream ID:", stream_id)
        print("Ingestion address:", ingestion.get("ingestionAddress"))
        print("Stream key:", ingestion.get("streamName"))
    except HttpError as err:
        print(f"API error ({err.resp.status}): {err}")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
