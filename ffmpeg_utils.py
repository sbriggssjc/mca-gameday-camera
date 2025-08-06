import logging
from typing import List, Optional

def build_ffmpeg_args(
    *,
    video_source: str,
    output_url: str,
    audio_device: Optional[str],
    audio_gain_db: float = 0.0,
    resolution: str = "1280x720",
    framerate: int = 30,
    video_codec: str = "libx264",
    video_is_pipe: bool = False,
    video_format: str = "v4l2",
    extra_args: Optional[List[str]] = None,
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
        Target resolution (e.g. ``"1280x720"``).
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
    """

    cmd: List[str] = ["ffmpeg", "-loglevel", "verbose", "-y"]

    if video_is_pipe:
        cmd += [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            resolution,
            "-framerate",
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

    cmd += [
        "-c:v",
        video_codec,
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        "4500k",
        "-maxrate",
        "6000k",
        "-bufsize",
        "6000k",
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

    cmd += ["-f", "flv", output_url]
    return cmd
