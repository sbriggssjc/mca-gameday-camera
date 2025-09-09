"""Encode frames to file or YouTube RTMP using ``ffmpeg``."""
from __future__ import annotations

import subprocess
from typing import Optional, Tuple


class Streamer:
    """Small wrapper around an ``ffmpeg`` subprocess."""

    def __init__(
        self,
        out_file: str | None = None,
        rtmp_url: str | None = None,
        rtmp_key: str | None = None,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        encoder: str = "h264_v4l2m2m",
        bitrate: str = "8000k",
        keyint: int = 60,
    ) -> None:
        width, height = resolution
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            encoder,
            "-b:v",
            bitrate,
            "-g",
            str(keyint),
        ]
        if out_file:
            cmd += [out_file]
        elif rtmp_url:
            url = rtmp_url.rstrip("/")
            if rtmp_key:
                url = f"{url}/{rtmp_key}"
            cmd += ["-f", "flv", url]
        else:  # pragma: no cover - sanity guard
            raise ValueError("either out_file or rtmp_url must be provided")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # ------------------------------------------------------------------
    def write(self, frame) -> None:
        """Write a single BGR frame to the encoder."""

        if self.proc.stdin:
            self.proc.stdin.write(frame.tobytes())

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()
