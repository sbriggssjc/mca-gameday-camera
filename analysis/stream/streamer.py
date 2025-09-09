"""Utilities for piping frames to an ``ffmpeg`` subprocess."""
from __future__ import annotations

import fcntl
import os
import subprocess
from typing import Optional, Tuple


class FrameToFFmpeg:
    """Send BGR frames to ``ffmpeg`` for recording or streaming.

    The process is fed by a ``stdin`` pipe using the ``rawvideo`` muxer.  The
    encoder is ``h264_v4l2m2m`` which is available on Jetson platforms.  Frames
    may change resolution dynamically; if so, the underlying ``ffmpeg`` process
    is restarted with the new width/height.
    """

    def __init__(
        self,
        *,
        out_file: Optional[str] = None,
        rtmp_url: Optional[str] = None,
        rtmp_key: Optional[str] = None,
        fps: int = 30,
        bitrate: str = "8000k",
        keyint: int = 60,
        encoder: str = "h264_v4l2m2m",
        resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        if not out_file and not rtmp_url:
            raise ValueError("either out_file or rtmp_url must be provided")

        self.out_file = out_file
        # Require both url and key for streaming; otherwise disable it
        if rtmp_url and rtmp_key:
            self.rtmp_url = f"{rtmp_url.rstrip('/')}/{rtmp_key}"
        else:
            self.rtmp_url = None
        self.fps = fps
        self.bitrate = bitrate
        self.keyint = keyint
        self.encoder = encoder
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._resolution: Optional[Tuple[int, int]] = None
        if resolution:
            self._open(*resolution)

    # ------------------------------------------------------------------
    def _buf_rate(self) -> str:
        """Return ~1.5x the target bitrate for maxrate/bufsize."""

        if self.bitrate.endswith("k"):
            base = float(self.bitrate[:-1])
            return f"{int(base * 1.5)}k"
        if self.bitrate.endswith("M"):
            base = float(self.bitrate[:-1])
            return f"{base * 1.5}M"
        # Fallback to original if unrecognized
        return self.bitrate

    # ------------------------------------------------------------------
    def _open(self, width: int, height: int) -> None:
        """Spawn the ffmpeg subprocess for the given resolution."""

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
            str(self.fps),
            "-i",
            "-",
            "-c:v",
            self.encoder,
            "-b:v",
            self.bitrate,
            "-g",
            str(self.keyint),
        ]

        if self.out_file:
            cmd += [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                self.out_file,
            ]
        else:
            rate = self._buf_rate()
            cmd += [
                "-tune",
                "zerolatency",
                "-preset",
                "fast",
                "-maxrate",
                rate,
                "-bufsize",
                rate,
                "-f",
                "flv",
                self.rtmp_url,
            ]

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._resolution = (width, height)

        # Make stdin non-blocking so that backpressure drops frames instead of
        # stalling the pipeline.
        if self._proc.stdin is not None:
            fd = self._proc.stdin.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    # ------------------------------------------------------------------
    def write(self, frame) -> None:
        """Write a single BGR frame to ffmpeg, dropping on backpressure."""

        h, w = frame.shape[:2]
        if self._proc is None or (w, h) != self._resolution:
            if self._proc:
                self.close()
            self._open(w, h)

        try:
            if self._proc and self._proc.stdin:
                self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, BlockingIOError):
            # Drop the frame if ffmpeg cannot keep up
            pass

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._proc:
            if self._proc.stdin:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            self._proc.wait()
            self._proc = None
            self._resolution = None


# Backwards compatibility for previous imports
Streamer = FrameToFFmpeg
