"""Utilities for piping frames to an ``ffmpeg`` subprocess."""
from __future__ import annotations

import os
import subprocess
import sys
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

        os.makedirs(os.path.dirname(self.path or "output"), exist_ok=True)
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._spawn(encoder=self.encoder)

    # ------------------------------------------------------------------
    def _build_cmd(self, encoder: str) -> list[str]:
        base_in = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",        # overwrite output files
            "-nostdin",  # avoid blocking on stdin
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
        ]
        enc = [
            "-an",
            "-c:v",
            encoder,
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            self.bitrate,
            "-maxrate",
            self.bitrate,
            "-bufsize",
            str(int(1.5 * int(self.bitrate.rstrip("k")))) + "k",
            "-g",
            str(self.keyint),
        ]
        ext = (os.path.splitext(self.path)[1] if self.path else "").lower()
        if self.stream:
            assert self.rtmp_url and self.rtmp_key, "RTMP url/key required"
            out = ["-f", "flv", f"{self.rtmp_url}/{self.rtmp_key}"]
        elif self.segment_seconds > 0:
            movflags = (
                "+frag_keyframe+empty_moov+separate_moof+default_base_moof"
                if self.fragmented_mp4
                else "+faststart"
            )
            out = [
                "-movflags",
                movflags,
                "-f",
                "segment",
                "-segment_time",
                str(self.segment_seconds),
                "-reset_timestamps",
                "1",
                "-strftime",
                "1",
                self.path,
            ]
        elif ext == ".mkv":
            out = ["-f", "matroska", self.path]
        elif ext == ".mp4" and self.fragmented_mp4:
            out = [
                "-movflags",
                "+frag_keyframe+empty_moov+separate_moof+default_base_moof",
                self.path,
            ]
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
        if not self._alive():
            if self.encoder != "libx264":
                print("[streamer] restarting with libx264", file=sys.stderr)
                self._restart_with("libx264")
                return
            raise BrokenPipeError("ffmpeg process is dead")

        try:
            assert self._proc and self._proc.stdin
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError):
            if self.encoder != "libx264":
                print("[streamer] fallback to libx264", file=sys.stderr)
                self._restart_with("libx264")
                return
            raise
        except BlockingIOError:
            return

    # ------------------------------------------------------------------
    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.stdin:
            try:
                proc.stdin.flush()
            finally:
                proc.stdin.close()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=1)
                except Exception:
                    pass


# Backwards compatibility
Streamer = FrameToFFmpeg
