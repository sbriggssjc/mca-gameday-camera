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
        Path to a V4L2 device, device index, video file or a GStreamer
        pipeline prefixed with ``"gst:"``.
    resolution:
        Desired ``(width, height)`` tuple.  Defaults to 4K.
    fps:
        Target frames-per-second.  Defaults to ``30``.
    fourcc:
        Requested pixel format.  Either ``"MJPG"`` or ``"YUYV"``.
    buffer_size:
        Number of frames to keep in the internal deque.
    backend:
        OpenCV capture backend.  ``"auto"`` selects ``CAP_V4L2`` for
        ``/dev/video*`` sources, ``"v4l2"`` forces V4L2 and ``"gst"`` uses
        GStreamer.  When using the GStreamer backend a source string may be
        prefixed with ``"gst:"`` to supply an explicit pipeline description.
    """

    def __init__(
        self,
        device: str | int,
        resolution: tuple[int, int] = (3840, 2160),
        fps: int = 30,
        fourcc: str = "MJPG",
        buffer_size: int = 8,
        backend: str = "auto",
    ) -> None:
        self.device = device
        self.backend = backend
        if (
            isinstance(device, str)
            and device.startswith("gst:")
            and backend == "auto"
        ):
            # Automatically switch to GST backend when an explicit pipeline
            # string is provided via "gst:<pipeline>".
            self.backend = "gst"
        self.resolution = resolution
        self.requested_fps = fps
        self.requested_fourcc = fourcc
        self.buffer: Deque[Tuple[float, "cv2.Mat"]] = deque(maxlen=buffer_size)
        self._fps_times: Deque[float] = deque(maxlen=128)
        self.stats = CameraStats()

        self.cap: Optional[cv2.VideoCapture] = None
        first_frame = self._open_capture()
        ts = time.time()
        if first_frame is None:
            raise RuntimeError("Failed to read initial frame")
        self.buffer.append((ts, first_frame))

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

    def _open_capture(self) -> Optional["cv2.Mat"]:

        # Explicit GStreamer pipeline input
        if isinstance(self.device, str) and self.device.startswith("gst:"):
            pipeline = self.device[4:]
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {self.device}")
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    "GStreamer pipeline opened but first frame read failed"
                )
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.resolution[0]
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.resolution[1]
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or float(
                self.requested_fps
            )
            fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            if fourcc_int:
                self.fourcc = "".join(
                    [chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]
                )
            else:
                self.fourcc = "GST"
            return frame

        # Regular V4L2/video file input
        flags = cv2.CAP_V4L2 if self._is_v4l2() else 0

        if self.backend == "gst":
            return self._open_gst_capture()

        flags = 0
        if self.backend == "v4l2" or (self.backend == "auto" and self._is_v4l2()):
            flags = cv2.CAP_V4L2

        self.cap = cv2.VideoCapture(self.device, flags)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.device}")

        ok, frame = self._configure_and_read(
            self.resolution[0],
            self.resolution[1],
            self.requested_fps,
            self.requested_fourcc,
        )
        if ok:
            return frame


        # Auto-fallback to GStreamer pipelines when V4L2 read fails
        if self._is_v4l2():
            self.cap.release()
            w, h = self.resolution
            fps = self.requested_fps
            pipelines = [
                (
                    "MJPEG",
                    (
                        f"v4l2src device={self.device} ! "
                        f"image/jpeg,framerate={fps}/1,width={w},height={h} ! "
                        "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
                        "appsink sync=false max-buffers=2 drop=true"
                    ),
                ),
                (
                    "H264",
                    (
                        f"v4l2src device={self.device} ! "
                        f"video/x-h264,stream-format=avc,framerate={fps}/1,width={w},height={h} ! "
                        "h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
                        "appsink sync=false max-buffers=2 drop=true"
                    ),
                ),
            ]
            for label, pipeline in pipelines:
                logging.info("Falling back to GStreamer %s pipeline", label)
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if not cap or not cap.isOpened():
                    continue
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.cap = cap
                    self.width, self.height, self.fps = w, h, float(fps)
                    self.fourcc = label
                    logging.info("Using GStreamer %s pipeline", label)

        if self.backend in ("auto", "v4l2") and self._is_v4l2():
            tried: List[str] = []
            for w, h, fps, fourcc in self._fallback_profiles():
                tried.append(f"{w}x{h}@{fps} {fourcc}")
                ok, frame = self._configure_and_read(w, h, fps, fourcc)
                if ok:
                    logging.info("Falling back to %s", tried[-1])
                    self.resolution = (w, h)
                    self.requested_fps = fps
                    self.requested_fourcc = fourcc

                    return frame
                cap.release()

        raise RuntimeError(
            "Camera opened but first frame read failed—check resolution/format."
        )

    def _open_gst_capture(self) -> Optional["cv2.Mat"]:
        device = self.device
        if isinstance(device, str) and device.startswith("gst:"):
            pipeline = device[4:]
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not cap or not cap.isOpened():
                raise RuntimeError(f"Cannot open GStreamer pipeline: {pipeline}")
            self.cap = cap
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("GStreamer pipeline opened but first frame read failed")
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = float(cap.get(cv2.CAP_PROP_FPS))
            self.fourcc = "GST"
            return frame

        if isinstance(device, str) and device.startswith("/dev/"):
            w, h = self.resolution
            fps = self.requested_fps
            mjpg = (
                f"v4l2src device={device} ! image/jpeg,framerate={fps}/1,width={w},height={h} "
                "! jpegdec ! videoconvert ! appsink"
            )
            h264 = (
                f"v4l2src device={device} ! video/x-h264,framerate={fps}/1,width={w},height={h} "
                "! h264parse ! avdec_h264 ! videoconvert ! appsink"
            )
            for pipe, fourcc in ((mjpg, "MJPG"), (h264, "H264")):
                cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                if not cap or not cap.isOpened():
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    continue
                self.cap = cap
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = float(cap.get(cv2.CAP_PROP_FPS))
                self.fourcc = fourcc
                logging.info(
                    "Negotiated %dx%d@%.2f %s via GStreamer",
                    self.width,
                    self.height,
                    self.fps,
                    self.fourcc,
                )
                return frame
            raise RuntimeError(f"Cannot open video source: {device}")

        cap = cv2.VideoCapture(device, cv2.CAP_GSTREAMER)
        if not cap or not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {device}")
        self.cap = cap
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("GStreamer pipeline opened but first frame read failed")
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.fourcc = "GST"
        return frame

    def _configure_and_read(
        self, w: int, h: int, fps: int, fourcc: str
    ) -> tuple[bool, Optional["cv2.Mat"]]:
        assert self.cap is not None
        # Apply settings: FOURCC, then size, then FPS
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Read negotiated values
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        logging.info(
            "Negotiated %dx%d@%.2f %s",
            self.width,
            self.height,
            self.fps,
            self.fourcc.strip(),
        )

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def _reader(self) -> None:
        fail_count = 0
        last_warn = 0.0
        while not self._stop:
            assert self.cap is not None
            ok, frame = self.cap.read()
            ts = time.time()
            if not ok or frame is None:
                fail_count += 1
                self.stats.dropped += 1
                if fail_count >= 5:
                    now = time.time()
                    if now - last_warn > 5.0:
                        logging.warning("Camera read failed %d times; retrying", fail_count)
                        last_warn = now
                    recovered = False
                    for _ in range(5):
                        self.cap.grab()
                        ok, frame = self.cap.read()
                        if ok and frame is not None:
                            recovered = True
                            break
                    if not recovered:
                        self._auto_retry()
                        fail_count = 0
                        continue
                    fail_count = 0
                    ts = time.time()
                else:
                    time.sleep(0.01)
                    continue
            else:
                fail_count = 0

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
            frame = self._open_capture()
            if frame is not None:
                self.buffer.clear()
                self.buffer.append((time.time(), frame))
            logging.info("Reconnected camera %s", self.device)
        except RuntimeError:
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

    def warmup(self, seconds: float = 1.0) -> bool:
        """Attempt to read frames for up to ``seconds`` seconds.

        Returns ``True`` if at least one frame was successfully read.
        """

        end = time.time() + seconds
        success = False
        while time.time() < end and self.is_open():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                success = True
            time.sleep(0.01)
        return success

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
    except RuntimeError:
        logging.error("Device %s not found", dev)
