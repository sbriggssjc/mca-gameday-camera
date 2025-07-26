import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def load_env(env_path: str = ".env") -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def build_ffmpeg_command(url: str, device: str = "/dev/video0") -> list[str]:
    return [
        "ffmpeg",
        "-f", "v4l2",
        "-framerate", "60",
        "-video_size", "1920x1080",
        "-i", device,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-b:v", "4500k",
        "-maxrate", "4500k",
        "-bufsize", "9000k",
        "-g", "120",
        "-c:a", "aac",
        "-b:a", "128k",
        "-f", "flv", url,
    ]


def main() -> None:
    load_env()
    url = os.environ.get("YOUTUBE_RTMP_URL")
    if not url:
        sys.exit("Missing YOUTUBE_RTMP_URL environment variable")

    log_dir = Path("livestream_logs")
    log_dir.mkdir(exist_ok=True)

    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"stream_{timestamp}.log"
        with log_file.open("w") as lf:
            process = subprocess.Popen(
                build_ffmpeg_command(url), stdout=lf, stderr=lf
            )
            ret = process.wait()
            lf.write(f"\nffmpeg exited with code {ret}\n")
        if ret == 0:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
