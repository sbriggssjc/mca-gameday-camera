import logging
import subprocess
import threading
from typing import List, Optional, Tuple


def detect_encoder(input_type: str | None = None) -> str:
    """Detect and return a usable H.264 encoder.

    Preference order:
    1. ``h264_v4l2m2m`` (Jetson hardware encoder)
    2. ``h264_nvmpi``
    3. ``libx264``

    ``h264_omx`` is intentionally skipped due to reliability issues. A
    ``RuntimeError`` is raised if no suitable encoder is found.

    When ``input_type`` is ``"image2pipe"`` (MJPEG frames piped via stdin),
    Jetson hardware encoders output an empty stream. In this case ``libx264`` is
    forced if available.
    """

    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        encoders = result.stdout
        if input_type == "image2pipe":
            print("⚠️ Forcing encoder to libx264 due to piped MJPEG input (image2pipe)")
            if "libx264" in encoders:
                return "libx264"
        else:
            if "h264_v4l2m2m" in encoders:
                return "h264_v4l2m2m"
            if "h264_nvmpi" in encoders:
                return "h264_nvmpi"
            if "libx264" in encoders:
                return "libx264"
    except Exception:
        pass
    raise RuntimeError(
        "❌ No usable H.264 encoder found (looked for h264_v4l2m2m, h264_nvmpi, libx264)."
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
    resolution: str = "426x240",
    framerate: int = 30,
    video_codec: str = "libx264",
    video_is_pipe: bool = False,
    video_format: str = "v4l2",
    preset: str = "veryfast",
    bitrate: str = "2000k",
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
        Target resolution (e.g. ``"426x240"``).
    framerate:
        Target frames per second.
    video_codec:
        Video encoder to use (defaults to ``libx264``).
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
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
        ]
    else:
        encoder_flags = ["-c:v", video_codec, "-pix_fmt", "yuv420p"]

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
