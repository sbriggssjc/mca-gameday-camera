import os, subprocess, sys, time, shlex, shutil
from urllib.parse import urlparse


def build_ffmpeg_cmd(
    pulse_source,
    rtmp_url,
    video_input=None,
    *,
    width=1280,
    height=720,
    fps=30,
    vbitrate="2500k",
    maxrate="3000k",
    bufsize="3000k",
):
    """Construct an FFmpeg command for streaming.

    If ``video_input`` is ``None`` a black video source is generated.
    ``pulse_source`` is *not* shell expanded; callers must supply the exact
    Pulse device name.
    """

    # Audio filter chain:
    #  - aformat to s16:48k
    #  - channelmap: if mono, duplicate to stereo
    #  - acompressor: light compression to lift quiet speech
    #  - alimiter: prevent clipping at 0dBFS
    af = [
        "aformat=sample_fmts=s16:sample_rates=48000",
        "channelmap=channel_layout=stereo",
        "acompressor=threshold=-18dB:ratio=3:attack=10:release=200:makeup=6",
        "alimiter=limit=0.9",
    ]

    audio_in = [
        "-f",
        "pulse",
        "-thread_queue_size",
        "1024",
        "-ac",
        "2",
        "-ar",
        "48000",
        "-i",
        pulse_source,
    ]

    # Video: if we have a v4l2 device or x11grab, add it; otherwise stream
    # audio-only with a black color source
    if video_input:
        video_in = [
            "-f",
            "v4l2",
            "-framerate",
            str(fps),
            "-video_size",
            f"{width}x{height}",
            "-i",
            video_input,
        ]
    else:
        video_in = [
            "-f",
            "lavfi",
            "-i",
            f"color=size={width}x{height}:rate={fps}:color=black",
        ]

    out = [
        # Encode
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-b:v",
        vbitrate,
        "-maxrate",
        maxrate,
        "-bufsize",
        bufsize,
        "-g",
        str(fps * 2),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ar",
        "48000",
        "-af",
        ",".join(af),
        # Low-latency/robustness
        "-tune",
        "zerolatency",
        "-flvflags",
        "no_duration_filesize",
        "-f",
        "flv",
        rtmp_url,
    ]

    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "warning",
    ] + audio_in + video_in + out


def stream_loop(pulse_source, rtmp_url, video_input=None, max_retries=100, backoff=5):
    tries = 0
    while True:
        cmd = build_ffmpeg_cmd(pulse_source, rtmp_url, video_input)
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        print(f"[ffmpeg] launching: {cmd_str}")
        try:
            rc = subprocess.call(cmd_str, shell=True)
        except FileNotFoundError:
            rc = 1
            print("[ffmpeg] executable not found")
        print(f"[ffmpeg] exited rc={rc}")
        tries += 1
        if tries >= max_retries:
            sys.exit(rc or 1)
        time.sleep(backoff)


def _normalize_youtube_url(url_or_key: str) -> str:
    """Return a canonical YouTube RTMP(S) ingest URL."""
    parsed = urlparse(url_or_key)
    scheme = parsed.scheme.lower()
    if scheme in {"rtmp", "rtmps"}:
        key = parsed.path.rsplit("/", 1)[-1]
    else:
        key = url_or_key
        scheme = "rtmps"
    host = "a.rtmps.youtube.com" if scheme == "rtmps" else "a.rtmp.youtube.com"
    return f"{scheme}://{host}/live2/{key}"


def _ensure_pulse_source(src: str) -> str:
    """Return ``src`` if it exists, otherwise ``default``."""
    if not src:
        return "default"
    try:
        if shutil.which("pactl"):
            out = subprocess.check_output(
                ["pactl", "list", "sources", "short"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            names = [line.split("\t", 2)[1] for line in out.splitlines()]
            if src in names:
                return src
    except Exception:
        pass
    return "default"


if __name__ == "__main__":
    src_env = os.environ.get("PULSE_SOURCE")
    url_env = (
        os.environ.get("YOUTUBE_RTMP_URL")
        or os.environ.get("YT_RTMP_URL")
        or os.environ.get("YOUTUBE_STREAM_KEY")
    )
    vid = os.environ.get("VIDEO_INPUT")  # optional, e.g., /dev/video0
    if not url_env:
        print("Missing YOUTUBE_RTMP_URL/YT_RTMP_URL/YOUTUBE_STREAM_KEY", file=sys.stderr)
        sys.exit(2)
    url = _normalize_youtube_url(url_env)
    src = _ensure_pulse_source(src_env)
    stream_loop(src, url, vid)
