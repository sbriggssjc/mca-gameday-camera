"""Video capture helpers for the gameday camera pipeline.

Two implementations are provided:

``Capture``
    A small historical wrapper that keeps a ring buffer of frames.  It is
    still used by some legacy scripts.

``FrameCapture``
    A more robust 4K capture class that negotiates the requested V4L2
    parameters, continuously reads frames on a background thread and
    automatically reconnects when the device stalls.

Example
-------
Run the module directly to print the negotiated camera capabilities::

    $ python -m analysis.camera.capture /dev/video0

The program will log the resolution, FPS and codec that were actually
negotiated with the device.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import threading
import time
from typing import Deque, List, Optional, Tuple

import cv2


class Capture:
    """4K video capture with an in-memory ring buffer."""

    def __init__(
        self,
        source: str | int,
        resolution: tuple[int, int] = (3840, 2160),
        fps: int = 30,
        buffer_seconds: float = 2.0,
    ) -> None:
        """Create a new capture device.

        Parameters
        ----------
        source:
            Path to a video file or V4L2 device (e.g. ``"/dev/video0"``).
        resolution:
            Desired ``(width, height)`` in pixels.
        fps:
            Target frames-per-second for the device.
        buffer_seconds:
            Duration of the internal ring buffer used for pre-roll.
        """

        self.source = source
        self.resolution = resolution
        self.fps = fps
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.buffer: Deque[Tuple[float, "cv2.Mat"]] = deque(
            maxlen=int(fps * buffer_seconds)
        )

    def read(self) -> tuple[bool, "cv2.Mat", float]:
        """Read a frame from the device.

        Returns a tuple ``(ok, frame, ts)`` where ``ts`` is a UNIX
        timestamp.  Successful frames are appended to the ring buffer.
        """

        ok, frame = self.cap.read()
        ts = time.time()
        if ok:
            self.buffer.append((ts, frame))
        return ok, frame, ts

    def get_buffer(self) -> List[Tuple[float, "cv2.Mat"]]:
        """Return a list copy of the buffered frames."""

        return list(self.buffer)

    def release(self) -> None:
        """Release the underlying ``VideoCapture`` object."""

        self.cap.release()


@dataclass
class CameraStats:
    """Simple runtime statistics for :class:`FrameCapture`.

    Attributes
    ----------
    fps:
        Estimated frames-per-second based on the last second of samples.
    dropped:
        Number of frames dropped because the internal buffer was full or the
        camera stalled.
    """

    fps: float = 0.0
    dropped: int = 0


class FrameCapture:
    """Robust frame capture with background thread and auto-retry.

    Parameters
    ----------
    device:
        Path to a V4L2 device, device index, or video file.
    resolution:
        Desired ``(width, height)`` tuple.  Defaults to 4K.
    fps:
        Target frames-per-second.  Defaults to ``30``.
    fourcc:
        Requested pixel format.  Either ``"MJPG"`` or ``"YUYV"``.
    buffer_size:
        Number of frames to keep in the internal deque.
    """

    def __init__(
        self,
        device: str | int,
        resolution: tuple[int, int] = (3840, 2160),
        fps: int = 30,
        fourcc: str = "MJPG",
        buffer_size: int = 8,
    ) -> None:
        self.device = device
        self.resolution = resolution
        self.requested_fps = fps
        self.requested_fourcc = fourcc
        self.buffer: Deque[Tuple[float, "cv2.Mat"]] = deque(maxlen=buffer_size)
        self._fps_times: Deque[float] = deque(maxlen=128)
        self.stats = CameraStats()

        self.cap: Optional[cv2.VideoCapture] = None
        self._open_capture()

        self._stop = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _is_v4l2(self) -> bool:
        return isinstance(self.device, int) or (
            isinstance(self.device, str) and self.device.startswith("/dev/video")
        )

    def _open_capture(self) -> None:
        flags = cv2.CAP_V4L2 if self._is_v4l2() else 0
        self.cap = cv2.VideoCapture(self.device, flags)
        if not self.cap or not self.cap.isOpened():
            logging.error("Unable to open capture device %s", self.device)
            raise FileNotFoundError(f"Capture device {self.device} not found")

        # Configure device if using V4L2
        if flags:
            w, h = self.resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
            self.cap.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.requested_fourcc)
            )

        # Negotiated values
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        if (self.width, self.height) != self.resolution:
            logging.warning(
                "Negotiated resolution %sx%s differs from request %s",
                self.width,
                self.height,
                self.resolution,
            )
        if round(self.fps) != self.requested_fps:
            logging.warning(
                "Negotiated FPS %.2f differs from request %s", self.fps, self.requested_fps
            )
        if self.fourcc.strip() != self.requested_fourcc:
            logging.warning(
                "Negotiated FOURCC %s differs from request %s",
                self.fourcc,
                self.requested_fourcc,
            )

    def _reader(self) -> None:
        while not self._stop:
            assert self.cap is not None
            ok, frame = self.cap.read()
            ts = time.time()
            if not ok or frame is None:
                logging.warning("Camera stall detected; attempting reconnect")
                self.stats.dropped += 1
                self._auto_retry()
                continue

            if len(self.buffer) == self.buffer.maxlen:
                self.stats.dropped += 1
            self.buffer.append((ts, frame))

            # update FPS statistics
            self._fps_times.append(ts)
            while self._fps_times and ts - self._fps_times[0] > 1.0:
                self._fps_times.popleft()
            self.stats.fps = float(len(self._fps_times))

    def _auto_retry(self) -> None:
        try:
            if self.cap:
                self.cap.release()
        except Exception:  # pragma: no cover - defensive
            pass
        time.sleep(1.0)
        try:
            self._open_capture()
            logging.info("Reconnected camera %s", self.device)
        except FileNotFoundError:
            logging.error("Failed to reconnect camera %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_open(self) -> bool:
        return bool(self.cap and self.cap.isOpened())

    def read(self) -> tuple[Optional["cv2.Mat"], float]:
        try:
            ts, frame = self.buffer[-1]
            return frame, ts
        except IndexError:
            return None, 0.0

    def warmup(self, seconds: float = 1.0) -> None:
        """Read and discard frames for a short period."""

        end = time.time() + seconds
        while time.time() < end and self.is_open():
            self.cap.read()
            time.sleep(0.01)

    def release(self) -> None:
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
    try:
        cam = FrameCapture(dev)
        cam.warmup(0.5)
        logging.info(
            "width=%s height=%s fps=%.2f fourcc=%s",
            cam.width,
            cam.height,
            cam.fps,
            cam.fourcc,
        )
    except FileNotFoundError:
        logging.error("Device %s not found", dev)
