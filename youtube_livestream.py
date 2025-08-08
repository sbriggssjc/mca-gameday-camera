import argparse
import argparse
import os
import platform
import subprocess
from datetime import datetime, timezone
from ffmpeg_utils import build_ffmpeg_args
from config import StreamConfig, load_config

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


def run_ffmpeg(cfg: StreamConfig, *, test: bool = False) -> None:
    """Launch FFmpeg using parameters from ``cfg``.

    Parameters
    ----------
    cfg:
        Streaming configuration containing destination and encoder settings.
    test:
        If True, only print the command that would be executed.
    """
    rtmp_url = cfg.stream_key
    system = platform.system()
    if system == "Linux":
        cmd = build_ffmpeg_args(
            video_source="/dev/video0",
            audio_device=cfg.mic_device,
            output_url=rtmp_url,
            audio_gain_db=cfg.gain_boost,
            resolution=cfg.resolution,
            framerate=cfg.fps,
            video_codec=cfg.encoder,
            preset=cfg.preset,
            bitrate=cfg.bitrate,
            maxrate=cfg.maxrate,
            bufsize=cfg.bufsize,
            force_ipv4=cfg.force_ipv4,
            extra_args=["-input_format", "yuyv422"],
        )
    elif system == "Windows":
        # Retain Windows-specific command using DirectShow devices.
        cmd = [
            "ffmpeg",
            "-f",
            "dshow",
            "-i",
            "video=Integrated Camera:audio=Microphone (USB)",
            "-c:v",
            cfg.encoder,
            "-preset",
            cfg.preset,
            "-b:v",
            cfg.bitrate,
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-f",
            "flv",
            rtmp_url,
        ]
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    print("[FFMPEG COMMAND]", " ".join(cmd))
    if test:
        return
    try:
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("❌ ffmpeg not found. Please install FFmpeg.")
        return
    try:
        _, err = process.communicate()
        if process.returncode != 0:
            err_text = err.decode("utf-8", errors="ignore") if err else ""
            if err_text:
                print(err_text)
            print(f"ffmpeg exited with code {process.returncode}")
    except Exception as exc:
        print(f"FFmpeg execution failed: {exc}")
        if process.stderr:
            err_text = process.stderr.read().decode("utf-8", errors="ignore")
            if err_text:
                print(err_text)


def test_audio_capture(device: str = "hw:1,0", filename: str = "test.wav", duration: int = 3) -> None:
    """Record a short clip using ``arecord`` to verify audio input."""
    try:
        cmd = ["arecord", "-D", device, "-f", "cd", "-d", str(duration), filename]
        print("[ARECORD COMMAND]", " ".join(cmd))
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        print(f"Audio recorded to {filename}")
    except FileNotFoundError:
        print("❌ arecord not found. Please install alsa-utils.")
    except subprocess.CalledProcessError as exc:
        err_text = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        if err_text:
            print(err_text)
        print("Failed to record audio", exc)


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
    parser.add_argument(
        "--test",
        action="store_true",
        help="Print the FFmpeg command without executing",
    )
    parser.add_argument(
        "--test-audio",
        action="store_true",
        help="Record a short audio clip and exit",
    )
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--resolution", default=None, help="Capture resolution WxH")
    parser.add_argument("--fps", type=int, default=None, help="Capture FPS")
    parser.add_argument("--mic_device", default=None, help="ALSA mic device")
    parser.add_argument("--gain_boost", type=float, default=None, help="Audio gain in dB")
    parser.add_argument("--encoder", default=None, help="Video encoder")
    parser.add_argument("--preset", default=None, help="Encoder preset")
    parser.add_argument("--bitrate", default=None, help="Target video bitrate")
    parser.add_argument("--maxrate", default=None, help="Max video bitrate")
    parser.add_argument("--bufsize", default=None, help="Encoder buffer size")
    args = parser.parse_args()
    cfg: StreamConfig = load_config(args.config, args)
    if args.test_audio:
        test_audio_capture(cfg.mic_device)
        return

    service = authenticate_youtube(reset_auth=args.reset_auth)

    try:
        stream_id, ingestion = create_stream(service, args.title)
        broadcast_id = create_broadcast(
            service, args.title, stream_id, args.start_time
        )
        print("Broadcast ID:", broadcast_id)
        print("Stream ID:", stream_id)
        address = ingestion.get("ingestionAddress")
        stream_key = ingestion.get("streamName")
        print("Ingestion address:", address)
        print("Stream key:", stream_key)
        if not (address and stream_key):
            raise RuntimeError("Missing ingestion address or stream key")
        cfg.stream_key = f"{address}/{stream_key}"
        run_ffmpeg(cfg, test=args.test)
    except HttpError as err:
        print(f"API error ({err.resp.status}): {err}")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
