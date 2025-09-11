"""Utilities for piping frames to an ``ffmpeg`` subprocess."""
from __future__ import annotations

import os
import sys
import subprocess
import shlex
import time
from typing import Optional


class FrameToFFmpeg:
    """Send BGR frames to ``ffmpeg`` for recording or streaming."""

    def __init__(
        self,
        path: str | None = None,
        out_file: str | None = None,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        encoder: str = "h264_v4l2m2m",
        bitrate: str = "12000k",
        keyint: int = 60,
        stream: bool = False,
        rtmp_url: str | None = None,
        rtmp_key: str | None = None,
        fragmented_mp4: bool = True,
        segment_seconds: int = 0,
    ) -> None:
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path or out_file or "out.mp4"
        self.width = int(width) - (int(width) % 2)
        self.height = int(height) - (int(height) % 2)
        self.fps = int(fps)
        self.encoder = encoder
        self.bitrate = str(bitrate)
        self.keyint = int(keyint)
        self.stream = bool(stream)
        self.rtmp_url = rtmp_url
        self.rtmp_key = rtmp_key
        self.fragmented_mp4 = bool(fragmented_mp4)
        self.segment_seconds = int(segment_seconds)
        self.allow_stream_fail = True  # keep recording if RTMP dies

        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._spawn(encoder=self.encoder)

    # ------------------------------------------------------------------
    def _build_cmd(self, encoder: str) -> list[str]:
        base_in = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "-",
        ]
        enc = [
            "-an", "-c:v", encoder, "-pix_fmt", "yuv420p",
            "-b:v", self.bitrate, "-maxrate", self.bitrate,
            "-bufsize", str(int(1.5 * int(self.bitrate.rstrip("k")))) + "k",
            "-g", str(self.keyint),
        ]

        if self.stream and self.path:
            tee_spec = (
                f"[onfail=ignore:f=flv]{self.rtmp_url}/{self.rtmp_key}" +
                f"|[f=matroska]{self.path}"
            )
            out = ["-f", "tee", tee_spec]
        elif self.stream:
            out = ["-f", "flv", f"{self.rtmp_url}/{self.rtmp_key}"]
        else:
            ext = (os.path.splitext(self.path)[1] if self.path else "").lower()
            if ext == ".mkv":
                out = ["-f", "matroska", self.path]
            else:
                out = ["-movflags", "+faststart", self.path]
        return base_in + enc + out

    # ------------------------------------------------------------------
    def _spawn(self, encoder: str) -> None:
        cmd = self._build_cmd(encoder)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self.encoder = encoder

    # ------------------------------------------------------------------
    def _restart_with(self, encoder: str) -> None:
        self.close()
        self._spawn(encoder)

    # ------------------------------------------------------------------
    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------
    def write(self, frame) -> None:
        if self._proc is None or self._proc.poll() is not None:
            # ffmpeg died — fall back smartly
            if self.stream and self.allow_stream_fail and self.path:
                # try local-file only
                print(
                    "[streamer] RTMP/tee failed; falling back to local file only",
                    file=sys.stderr,
                )
                self.stream = False
                self._restart_with(
                    self.encoder if self.encoder != "h264_v4l2m2m" else "libx264"
                )
                return  # drop this frame
            # try encoder fallback once
            if self.encoder != "libx264":
                print("[streamer] restarting with libx264", file=sys.stderr)
                self._restart_with("libx264")
                return
            raise BrokenPipeError("ffmpeg process is dead")

        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError):
            # Same fallback logic on write failure
            if self.stream and self.allow_stream_fail and self.path:
                print(
                    "[streamer] write failed; switching to local file only",
                    file=sys.stderr,
                )
                self.stream = False
                self._restart_with(
                    self.encoder if self.encoder != "h264_v4l2m2m" else "libx264"
                )
                return
            if self.encoder != "libx264":
                print("[streamer] fallback to libx264", file=sys.stderr)
                self._restart_with("libx264")
                return
            raise
        except BlockingIOError:
            return  # drop frame to keep latency

    # ------------------------------------------------------------------
    def close(self) -> None:
        if not self._proc:
            return
        try:
            if self._proc.stdin:
                try:
                    self._proc.stdin.flush()
                except Exception:
                    pass
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            self._proc.wait(timeout=3)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None


# Backwards compatibility
Streamer = FrameToFFmpeg
