"""Utilities for piping frames to an ``ffmpeg`` subprocess."""
from __future__ import annotations

import fcntl
import logging
import os
import subprocess
from typing import Optional


logger = logging.getLogger(__name__)


class FrameToFFmpeg:
    """Send BGR frames to ``ffmpeg`` for recording or streaming."""

    def __init__(
        self,
        # output target
        path: str | None = None,
        out_file: str | None = None,  # backward compat
        # geometry / timing
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        # encoding
        encoder: str = "h264_v4l2m2m",
        bitrate: str = "12000k",
        keyint: int = 60,
        # streaming
        stream: bool = False,
        rtmp_url: str | None = None,
        rtmp_key: str | None = None,
    ) -> None:
        # normalize target name
        self.path = path or out_file or "out.mp4"  # accept either
        self.stream = bool(stream)
        self.rtmp_url = (rtmp_url or "").rstrip("/")
        self.rtmp_key = rtmp_key or ""
        self.width = int(width) - (int(width) % 2)
        self.height = int(height) - (int(height) % 2)
        self.fps = int(fps)
        self.encoder = encoder
        self.bitrate = bitrate
        self.keyint = int(keyint)

        self.cmd: list[str] = []
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._first_write = True

        self._build_cmd()
        self._start()

    # ------------------------------------------------------------------
    def _build_cmd(self) -> None:
        buf = str(int(1.5 * int(self.bitrate.rstrip("k")))) + "k"
        self.cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
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
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(self.keyint),
            "-c:v",
            self.encoder,
            "-b:v",
            self.bitrate,
            "-maxrate",
            self.bitrate,
            "-bufsize",
            buf,
        ]
        if self.stream:
            self.cmd += ["-f", "flv", f"{self.rtmp_url}/{self.rtmp_key}"]
        else:
            self.cmd += ["-movflags", "+faststart", self.path]

    # ------------------------------------------------------------------
    def _start(self) -> None:
        self._proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if self._proc.stdin is not None:
            fd = self._proc.stdin.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        if self._proc.stderr is not None:
            fd = self._proc.stderr.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        self._first_write = True

    # ------------------------------------------------------------------
    def _restart_with_encoder(self, new_encoder: str) -> None:
        self.close()
        self.encoder = new_encoder
        self._build_cmd()
        logger.warning("[streamer] fallback to libx264")
        self._start()

    # ------------------------------------------------------------------
    def write(self, frame) -> None:
        if not self._proc or not self._proc.stdin:
            return
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError):
            if self.encoder != "libx264":
                self._restart_with_encoder("libx264")
                return
            raise
        except BlockingIOError:
            return

        if self._first_write:
            err = ""
            if self._proc.stderr is not None:
                try:
                    err = self._proc.stderr.read().decode("utf-8", "ignore")
                except Exception:
                    pass
            rc = self._proc.poll()
            if (rc and rc != 0) or "h264_v4l2m2m" in err:
                if self.encoder != "libx264":
                    self._restart_with_encoder("libx264")
                    try:
                        if self._proc and self._proc.stdin:
                            self._proc.stdin.write(frame.tobytes())
                    except Exception:
                        pass
            self._first_write = False

    # ------------------------------------------------------------------
    def close(self) -> None:
        if not self._proc:
            return
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
        try:
            self._proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            try:
                self._proc.wait(timeout=1)
            except Exception:
                pass
        self._proc = None


# Backwards compatibility
Streamer = FrameToFFmpeg

