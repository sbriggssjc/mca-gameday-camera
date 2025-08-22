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
    input_format="mjpeg",
    video_delay=0.0,
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
    #  - highpass: roll off low-frequency rumble
    #  - volume: reduce overall level before compression
    #  - acompressor: gentle compression to lift quiet speech
    #  - alimiter: prevent clipping below full scale
    af = [
        "highpass=f=100",
        "volume=-3dB",
        "acompressor=threshold=-22dB:ratio=2.5:attack=12:release=250:makeup=2",
        "alimiter=limit=0.85",
    ]

    audio_in = [
        "-thread_queue_size",
        "4096",
        "-f",
        "pulse",
        "-ac",
        "2",
        "-ar",
        "48000",
        "-i",
        pulse_source,
    ]

    # Prefer a real video device; fall back to a generated black frame
    if video_input:
        video_in = [
            "-thread_queue_size",
            "4096",
            "-f",
            "v4l2",
            "-rtbufsize",
            "256M",
            "-input_format",
            input_format,
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

    filter_complex = (
        f"[1:v]setpts=PTS+{video_delay}/TB[v];"
        "[0:a]aresample=async=1:first_pts=0,asetpts=N/SR/TB[a]"
    )

    out = [
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-r",
        str(fps),
        "-vsync",
        "cfr",
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
        "-fflags",
        "+genpts",
    ] + audio_in + video_in + out


def stream_loop(
    pulse_source,
    rtmp_url,
    video_input=None,
    video_delay=0.0,
    *,
    width=1280,
    height=720,
    fps=30,
    input_format="mjpeg",
    max_retries=100,
    backoff=5,
):
    tries = 0
    while True:
        cmd = build_ffmpeg_cmd(
            pulse_source,
            rtmp_url,
            video_input,
            width=width,
            height=height,
            fps=fps,
            input_format=input_format,
            video_delay=video_delay,
        )
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
    # Prefer a real V4L2 device; fallback to black frame if missing
    dev = (
        os.environ.get("VIDEO_DEV")
        or os.environ.get("VIDEO_INPUT")
        or "/dev/video0"
    )
    if os.path.exists(dev):
        vid = dev
    else:
        print(f"[warn] {dev} not found; using black frame")
        vid = None

    # Resolution/FPS can be overridden via env (RES=WIDTHxHEIGHT, FPS=30)
    res = os.environ.get("RES", "1280x720")
    try:
        width_str, height_str = res.split("x", 1)
        width_val, height_val = int(width_str), int(height_str)
    except ValueError:
        width_val, height_val = 1280, 720
    fps_val = int(os.environ.get("FPS", "30"))
    # Camera format (try yuyv422 if MJPEG is unstable; use h264 if supported)
    fmt_val = os.environ.get("V4L2_FMT", "mjpeg")
    # Delay video to let audio "catch up" (tune in 0.05s steps)
    vid_delay_val = float(os.environ.get("VID_DELAY", "0.25"))
    if not url_env:
        print("Missing YOUTUBE_RTMP_URL/YT_RTMP_URL/YOUTUBE_STREAM_KEY", file=sys.stderr)
        sys.exit(2)
    url = _normalize_youtube_url(url_env)
    src = _ensure_pulse_source(src_env)
    stream_loop(
        src,
        url,
        vid,
        video_delay=vid_delay_val,
        width=width_val,
        height=height_val,
        fps=fps_val,
        input_format=fmt_val,
    )
