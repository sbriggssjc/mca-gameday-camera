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

        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._spawn(encoder=self.encoder)

    # ------------------------------------------------------------------
    def _normalize_rtmp(self) -> str | None:
        """Normalize RTMP(S) URL and append key."""
        # Accept rtmp:// or rtmps://; prefer RTMPS (YouTube often blocks 1935).
        url = (self.rtmp_url or "").strip()
        key = (self.rtmp_key or "").strip()
        if not url:
            return None
        # If user included the key in the URL, allow it as-is.
        if (
            url.startswith(("rtmp://", "rtmps://"))
            and url.rstrip("/").split("/")[-2:]
            and key
            and url.endswith("/" + key)
        ):
            return url  # full URL with key
        # If URL doesn't include scheme, add RTMPS
        if not url.startswith(("rtmp://", "rtmps://")):
            url = "rtmps://" + url.lstrip("/")
        # Force YouTube hostnames to rtmps://
        url = url.replace("rtmp://a.rtmp.youtube.com", "rtmps://a.rtmps.youtube.com")
        # Compose final
        if key and not url.rstrip("/").endswith("/" + key):
            url = url.rstrip("/") + "/" + key
        return url

    # ------------------------------------------------------------------
    def _build_cmd(self, encoder: str) -> list[str]:
        import os

        rtmp_full = self._normalize_rtmp()
        base_in = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
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
        if self.stream and self.path and rtmp_full:
            # Stream + record simultaneously via tee; ignore RTMP failure so local file continues.
            # NOTE: f=flv for RTMP(S); f=matroska for local MKV.
            tee_spec = f"[onfail=ignore:f=flv]{rtmp_full}|[f=matroska]{self.path}"
            out = ["-f", "tee", tee_spec]
        elif self.stream and rtmp_full:
            out = ["-f", "flv", rtmp_full]
        else:
            ext = (os.path.splitext(self.path)[1] if self.path else "").lower()
            out = ["-f", "matroska", self.path] if ext == ".mkv" else ["-movflags", "+faststart", self.path]
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
            if self.stream and self.path:
                # Drop stream, keep file
                print(
                    "[streamer] RTMP/tee failed; falling back to local file only",
                    file=sys.stderr,
                )
                self.stream = False
                self._restart_with(
                    self.encoder if self.encoder != "h264_v4l2m2m" else "libx264"
                )
                return
            if self.encoder != "libx264":
                self._restart_with("libx264")
                return
            raise BrokenPipeError("ffmpeg process is dead")
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError):
            if self.stream and self.path:
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
                self._restart_with("libx264")
                return
            raise
        except BlockingIOError:
            return

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


class MultiSinkStreamer:
    def __init__(self, path, width, height, fps, encoder, bitrate, keyint, rtmp_full):
        import os, subprocess
        self.width, self.height, self.fps = width, height, fps
        self.encoder, self.bitrate, self.keyint = encoder, bitrate, keyint
        self.path = path
        self.rtmp = rtmp_full
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        base = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", encoder, "-pix_fmt", "yuv420p",
            "-b:v", bitrate, "-maxrate", bitrate,
            "-bufsize", f"{int(1.5*int(bitrate.rstrip('k')))}k",
            "-g", str(keyint),
        ]
        # Local file sink (MKV for resilience)
        self.p_file = subprocess.Popen(
            [*base, "-f", "matroska", path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        # RTMP(S) sink
        self.p_rtmp = subprocess.Popen(
            [*base, "-f", "flv", rtmp_full],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

    def write(self, frame):
        # Try both sinks; if one dies, keep the other
        data = frame.tobytes()
        for p in (self.p_file, self.p_rtmp):
            if p and p.poll() is None:
                try:
                    p.stdin.write(data)
                except Exception:
                    pass  # drop for this sink

    def close(self):
        for p in (self.p_file, self.p_rtmp):
            if not p: continue
            try:
                if p.stdin:
                    try: p.stdin.flush()
                    except: pass
                    try: p.stdin.close()
                    except: pass
                p.wait(timeout=3)
            except Exception:
                try: p.kill()
                except: pass


# Backwards compatibility
Streamer = FrameToFFmpeg
