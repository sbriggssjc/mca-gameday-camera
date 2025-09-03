import logging
import subprocess
import threading
import os
import shlex
import time
from datetime import datetime
from typing import List, Optional, Tuple


def _sanity_probe(name: str) -> bool:
    """Return ``True`` if ``ffmpeg`` can open the given encoder."""

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=1280x720:rate=30",
        "-t",
        "1",
        "-c:v",
        name,
        "-f",
        "null",
        "-",
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False


def detect_encoder(
    preferred: Optional[List[str]] = None,
    input_type: str | None = None,
) -> str:
    """Detect and return a usable H.264 encoder.

    ``preferred`` may specify a custom encoder search order.  When ``input_type``
    is ``"image2pipe"`` (piped MJPEG frames), hardware encoders are skipped in
    favour of ``libx264``.
    """

    try:
        encoders = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"], text=True
        )
    except Exception:
        encoders = ""

    # Build a list of available H.264 encoders reported by ffmpeg
    available: List[str] = []
    for line in encoders.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        name = parts[1]
        if name.startswith("h264") or name == "libx264":
            available.append(name)

    # When piping frames, prefer software encoding to avoid unsupported hw paths
    if input_type == "image2pipe":
        if "libx264" in available and _sanity_probe("libx264"):
            return "libx264"

    # Determine search order.  Include common hardware encoders then libx264.
    common = ["h264_v4l2m2m", "h264_nvmpi", "h264_nvenc", "h264_omx", "h264_vaapi", "libx264"]
    candidates = preferred or common

    for name in candidates:
        if name in available and _sanity_probe(name):
            return name

    # Fallback: try any other reported encoder
    for name in available:
        if _sanity_probe(name):
            return name

    if _sanity_probe("libx264"):
        return "libx264"

    raise RuntimeError(
        "❌ No usable H.264 encoder found (none of the h264 encoders reported by ffmpeg worked).",
    )

def run_ffmpeg_command(cmd: List[str], timeout: int = 15) -> Tuple[int, str, str]:
    """Run an FFmpeg command with realtime stderr streaming.

    Parameters
    ----------
    cmd: List[str]
        Command and arguments to execute.
    timeout: int
        Maximum number of seconds to allow the process to run.

    Returns
    -------
    Tuple[int, str, str]
        A tuple of ``(returncode, stdout, stderr)``.
        ``stderr`` is fully captured even while being streamed.
    """

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stderr_lines: List[str] = []

    def _read_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_lines.append(line)
            logging.error("[ffmpeg] %s", line.rstrip())

    thread = threading.Thread(target=_read_stderr, daemon=True)
    thread.start()
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        thread.join()
        stderr = "".join(stderr_lines)
        logging.error("FFmpeg command timed out")
        logging.error(stderr)
        return -1, "", stderr
    thread.join()
    stderr = "".join(stderr_lines)
    if process.returncode != 0:
        logging.error("FFmpeg exited with code %s", process.returncode)
        logging.error(stderr)
    return process.returncode, stdout, stderr

def build_ffmpeg_args(
    *,
    video_source: str,
    output_url: str,
    audio_device: Optional[str],
    audio_gain_db: float = 0.0,
    resolution: str = "640x480",
    framerate: int = 30,
    video_codec: str = "h264_v4l2m2m",
    video_is_pipe: bool = False,
    video_format: str = "v4l2",
    preset: str = "veryfast",
    bitrate: str = "2500k",
    maxrate: str = "3000k",
    bufsize: str = "4000k",
    gop: int = 60,
    keyint_min: int = 30,
    local_record: Optional[str] = None,
    force_ipv4: bool = False,
    extra_args: Optional[List[str]] = None,
    diagnose_only: bool = False,
) -> List[str]:
    """Return a standardized FFmpeg command.

    Parameters
    ----------
    video_source:
        Path or identifier for the video input. Use "-" when piping raw frames.
    output_url:
        Destination URL or file path.
    audio_device:
        Identifier for the audio capture device. If ``None``, audio input is skipped
        and a log message is emitted.
    audio_gain_db:
        Gain to apply via the ``volume`` filter in decibels.
    resolution:
        Target resolution (e.g. ``"640x480"``).
    framerate:
        Target frames per second.
    video_codec:
        Video encoder to use (defaults to ``h264_v4l2m2m``).
    video_is_pipe:
        If True, treat ``video_source`` as raw frames on stdin.
    video_format:
        Input format when ``video_is_pipe`` is False (default ``v4l2``).
    extra_args:
        Additional FFmpeg arguments to append before the output target.
    force_ipv4:
        If True, append ``-rtmp_flags prefer_ipv4`` to prefer IPv4 RTMP.
    diagnose_only:
        When True, direct output to ``null`` for a non-networked dry run.
    """

    cmd: List[str] = ["ffmpeg", "-loglevel", "verbose", "-y"]

    if video_is_pipe:
        cmd += [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            resolution,
            "-r",
            str(framerate),
            "-i",
            "-",
        ]
    else:
        cmd += [
            "-f",
            video_format,
            "-framerate",
            str(framerate),
            "-video_size",
            resolution,
            "-i",
            video_source,
        ]

    if audio_device:
        cmd += [
            "-thread_queue_size",
            "512",
            "-f",
            "alsa",
            "-ac",
            "1",
            "-ar",
            "44100",
            "-i",
            audio_device,
        ]
    else:
        logging.info("Audio capture intentionally skipped")

    if video_codec == "libx264":
        encoder_flags = [
            "-vf",
            "format=yuv420p,setsar=1",
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-tune",
            "zerolatency",
        ]
    else:
        encoder_flags = ["-vf", "format=yuv420p,setsar=1", "-c:v", video_codec]

    cmd += encoder_flags + [
        "-b:v",
        bitrate,
        "-maxrate",
        maxrate,
        "-bufsize",
        bufsize,
        "-g",
        str(gop),
        "-keyint_min",
        str(keyint_min),
    ]

    if audio_device:
        cmd += [
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "44100",
            "-ac",
            "1",
            "-af",
            f"volume={audio_gain_db}dB",
        ]

    if extra_args:
        cmd += list(extra_args)

    if force_ipv4:
        cmd += ["-rtmp_flags", "prefer_ipv4"]

    if diagnose_only:
        cmd += ["-f", "null", "-"]
    elif local_record:
        out_spec = f"[f=flv:onfail=ignore]{output_url}|{local_record}"
        cmd += ["-f", "tee", out_spec]
    else:
        cmd += ["-f", "flv", output_url]
    return cmd

def build_stream_command(
    stream_key: str,
    *,
    video_device: str = "/dev/video0",
    resolution: str = "1280x720",
    framerate: int = 30,
    input_format: str = "mjpeg",
    encoder: str = "h264_v4l2m2m",
    audio_backend: str = "pulse",
    record_path: str = "recordings/raw/%Y%m%d_%H%M%S.mkv",
) -> List[str]:
    """Build the FFmpeg command for streaming and local recording."""
    video_in = [
        "-f",
        "v4l2",
        "-thread_queue_size",
        "8192",
        "-framerate",
        str(framerate),
        "-video_size",
        resolution,
        "-input_format",
        input_format,
        "-i",
        video_device,
    ]
    if audio_backend == "pulse":
        audio_in = ["-f", "pulse", "-thread_queue_size", "8192", "-i", "default"]
    else:
        audio_in = ["-f", "alsa", "-thread_queue_size", "8192", "-i", "plughw:1,0"]
    common = [
        "-filter:a",
        "aresample=async=1:min_hard_comp=0.100:first_pts=0",
        "-use_wallclock_as_timestamps",
        "1",
        "-fflags",
        "+genpts",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]
    if encoder == "h264_v4l2m2m":
        v_flags = [
            "-c:v",
            "h264_v4l2m2m",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "4500k",
            "-maxrate",
            "5000k",
            "-bufsize",
            "10000k",
            "-g",
            "60",
            "-sc_threshold",
            "0",
        ]
    else:
        v_flags = [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "4500k",
            "-maxrate",
            "5000k",
            "-bufsize",
            "10000k",
            "-g",
            "60",
            "-x264-params",
            "keyint=60:min-keyint=60:scenecut=0",
        ]
    a_flags = [
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
        "-ar",
        "48000",
    ]
    out_spec = (
        f"[f=flv:onfail=ignore]rtmps://a.rtmp.youtube.com/live2/{stream_key}"
        f"|[f=matroska]{record_path}"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-stats_period",
        "2",
    ] + video_in + audio_in + common + v_flags + a_flags + ["-f", "tee", out_spec]


def launch_ffmpeg(stream_key: str, *, dry_run: bool = False) -> subprocess.Popen | None:
    """Launch FFmpeg with automatic fallbacks.

    Returns the running ``Popen`` instance or ``None`` in ``dry_run`` mode.
    """
    attempts = [
        ("mjpeg", "h264_v4l2m2m", "pulse"),
        ("mjpeg", "h264_v4l2m2m", "alsa"),
        ("yuyv422", "h264_v4l2m2m", "pulse"),
        ("yuyv422", "h264_v4l2m2m", "alsa"),
        ("mjpeg", "libx264", "pulse"),
        ("mjpeg", "libx264", "alsa"),
        ("yuyv422", "libx264", "pulse"),
        ("yuyv422", "libx264", "alsa"),
    ]
    for fmt, enc, audio in attempts:
        cmd = build_stream_command(stream_key, input_format=fmt, encoder=enc, audio_backend=audio)
        logging.info("FFmpeg command: %s", shlex.join(cmd))
        if dry_run:
            print(shlex.join(cmd))
            return None
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        last = time.time()
        try:
            assert process.stderr is not None
            for line in process.stderr:
                logging.error("[ffmpeg] %s", line.rstrip())
                if "frame=" in line:
                    last = time.time()
                if any(err in line for err in ["Device or resource busy", "Input queue full", "Failed to open" ]):
                    raise RuntimeError("device error")
                if time.time() - last > 5:
                    raise RuntimeError("stalled")
            rc = process.wait()
            if rc == 0:
                return process
        except Exception as exc:
            process.kill()
            logging.warning("FFmpeg failed with %s", exc)
    return None
