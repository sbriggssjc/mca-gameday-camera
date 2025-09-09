"""Video capture with simple ring buffer.

This module provides a thin wrapper around ``cv2.VideoCapture`` that
configures a device for 4K capture and keeps a ring buffer of recent
frames.  The ring buffer is used to implement pre/post-roll recording in
the live pipeline.
"""
from __future__ import annotations

from collections import deque
import time
from typing import Deque, List, Tuple

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
