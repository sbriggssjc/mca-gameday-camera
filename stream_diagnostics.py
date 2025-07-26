import os
import shutil
import subprocess
from pathlib import Path


def load_env(env_path: str = ".env") -> None:
    """Load environment variables from a simple KEY=VALUE file."""
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg is not installed or not in PATH")
    return path


def test_stream(url: str | None = None, *, duration: int = 5, log_path: str = "test_stream.log") -> None:
    """Send a short test pattern to the RTMP endpoint to verify connectivity."""
    if url is None:
        load_env()
        url = os.environ.get("YOUTUBE_RTMP_URL")
    if not url:
        raise RuntimeError("Missing YOUTUBE_RTMP_URL environment variable")

    ensure_ffmpeg()

    log_file = Path(log_path)
    with log_file.open("w", encoding="utf-8", errors="ignore") as log:
        # Ping YouTube to check network connectivity
        log.write("Pinging youtube.com...\n")
        ping = subprocess.run([
            "ping",
            "-c",
            "3",
            "youtube.com",
        ], capture_output=True)
        stdout = ping.stdout.decode("utf-8", errors="ignore")
        stderr = ping.stderr.decode("utf-8", errors="ignore")
        log.write(stdout)
        log.write(stderr)
        if ping.returncode != 0:
            raise RuntimeError("Ping to youtube.com failed")

        log.write("\nStarting ffmpeg test stream...\n")
        cmd = [
            ensure_ffmpeg(),
            "-re",
            "-f", "lavfi",
            "-i", "testsrc=size=1280x720:rate=30",
            "-t", str(duration),
            "-f", "flv",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True)
        stdout = result.stdout.decode("utf-8", errors="ignore")
        stderr = result.stderr.decode("utf-8", errors="ignore")
        log.write(stdout)
        log.write(stderr)
        if result.returncode != 0:
            raise RuntimeError("ffmpeg test stream failed")
        if "frame=" not in result.stderr:
            raise RuntimeError("ffmpeg did not appear to send frames")

