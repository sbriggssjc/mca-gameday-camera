"""Utilities for piping frames to an ``ffmpeg`` subprocess."""
from __future__ import annotations

import fcntl
import logging
import os
import subprocess
from typing import Optional


logger = logging.getLogger(__name__)


class FrameToFFmpeg:
    """Send BGR frames to ``ffmpeg`` for recording or streaming.

    Frames are pushed to ``ffmpeg`` via ``stdin`` using the ``rawvideo`` muxer.
    The encoder defaults to ``h264_v4l2m2m`` but will automatically fall back
    to ``libx264`` if the hardware encoder is unavailable or misconfigured.
    """

    def __init__(
        self,
        *,
        out_file: Optional[str] = None,
        rtmp_url: Optional[str] = None,
        rtmp_key: Optional[str] = None,
        width: int,
        height: int,
        fps: int = 30,
        encoder: str = "h264_v4l2m2m",
        bitrate: str = "8000k",
        keyint: int = 60,
    ) -> None:
        if not out_file and not rtmp_url:
            raise ValueError("either out_file or rtmp_url must be provided")

        self.out_file = out_file
        # Require both url and key for streaming; otherwise disable it
        if rtmp_url and rtmp_key:
            self.rtmp_url = f"{rtmp_url.rstrip('/')}/{rtmp_key}"
        else:
            self.rtmp_url = None

        self.width = width
        self.height = height
        self.fps = fps
        self.encoder = encoder
        self.bitrate = bitrate
        self.keyint = keyint

        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._first_write = False
        self._open()

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
    def _open(self) -> None:
        """Spawn the ffmpeg subprocess."""

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
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            self.encoder,
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            self.bitrate,
            "-maxrate",
            self.bitrate,
            "-bufsize",
            self._buf_rate(),
            "-g",
            str(self.keyint),
        ]

        if self.out_file:
            cmd += ["-movflags", "+faststart", self.out_file]
        else:
            cmd += ["-tune", "zerolatency", "-preset", "fast", "-f", "flv", self.rtmp_url]

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._first_write = True

        if self._proc.stdin is not None:
            fd = self._proc.stdin.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        if self._proc.stderr is not None:
            fd = self._proc.stderr.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    # ------------------------------------------------------------------
    def write(self, frame) -> None:
        """Write a single BGR frame to ffmpeg, dropping on backpressure."""

        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            # Detect size changes; restart with the new resolution.
            self.close()
            self.width, self.height = w, h
            self._open()

        if not self._proc or not self._proc.stdin:
            return

        try:
            self._proc.stdin.write(frame.tobytes())
        except BlockingIOError:
            # Drop frame to keep latency under control
            return
        except BrokenPipeError:
            err = ""
            if self._proc.stderr is not None:
                try:
                    err = self._proc.stderr.read().decode("utf-8", "ignore")
                except Exception:
                    pass
            self._handle_failure(err, frame)
            return

        if self._first_write:
            err = ""
            if self._proc.stderr is not None:
                try:
                    err = self._proc.stderr.read().decode("utf-8", "ignore")
                except Exception:
                    pass
            rc = self._proc.poll()
            if (
                (rc and rc != 0)
                or "could not find a valid device" in err
                or "can't configure encoder" in err
            ):
                self._handle_failure(err, frame)
                return
            self._first_write = False

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._proc:
            if self._proc.stdin:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            if self._proc.stderr:
                try:
                    self._proc.stderr.close()
                except Exception:
                    pass
            self._proc.wait()
            self._proc = None
            self._first_write = False

    # ------------------------------------------------------------------
    def _handle_failure(self, err: str, frame) -> None:
        """Handle encoder failures and attempt a software fallback."""

        if self.encoder == "h264_v4l2m2m":
            logger.warning("fallback to libx264")
            self.close()
            self.encoder = "libx264"
            self._open()
            if self._proc and self._proc.stdin:
                try:
                    self._proc.stdin.write(frame.tobytes())
                except BlockingIOError:
                    return
        else:
            self.close()


# Backwards compatibility for previous imports
Streamer = FrameToFFmpeg
