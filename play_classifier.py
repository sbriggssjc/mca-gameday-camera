"""Play classifier using YOLOv5.

This module provides a simple wrapper over a YOLOv5 model to detect
football play events such as a touchdown (player moving quickly toward
the bottom of the frame) or a player exiting the frame at high speed.

The implementation relies on either OpenCV or PyTorch for image
processing. The model is expected to be a standard YOLOv5 model
trained to detect players. It can be loaded from a local path or via
``torch.hub``.
"""

from typing import List, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


class PlayClassifier:
    """Classifies sequences of frames for touchdown-like events."""

    def __init__(self, model_path: str = "yolov5s.pt", device: str = "cpu") -> None:
        """Load a YOLOv5 model.

        Parameters
        ----------
        model_path:
            Path to the YOLOv5 model weights. ``torch.hub`` will be used
            to fetch the model if ``model_path`` points to a known name
            like ``yolov5s.pt``.
        device:
            Device string understood by ``torch`` (e.g. ``"cpu"`` or
            ``"cuda"``).
        """
        if torch is None:
            raise ImportError("PyTorch is required for PlayClassifier")

        # load model via torch.hub (from ultralytics) or local path
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=model_path,
            force_reload=False,
            trust_repo=True,
        ).to(device)
        self.model.eval()
        self.device = device

    def _detect_player(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Run YOLOv5 on a single frame and return the largest box.

        Returns
        -------
        (x1, y1, x2, y2):
            Coordinates of the detected player's bounding box. If no
            player is detected, ``(-1, -1, -1, -1)`` is returned.
        """
        if frame is None:
            return (-1, -1, -1, -1)

        # convert BGR frame to RGB as expected by YOLO
        img = frame[:, :, ::-1]
        results = self.model(img)
        if results is None or len(results.xyxy[0]) == 0:
            return (-1, -1, -1, -1)

        boxes = results.xyxy[0][:, :4].cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        return tuple(map(int, boxes[idx]))  # type: ignore

    def classify_sequence(
        self,
        frames: List[np.ndarray],
        speed_threshold: float = 20.0,
        edge_fraction: float = 0.1,
    ) -> List[Tuple[str, int, Tuple[int, int]]]:
        """Analyze frames and detect events.

        Parameters
        ----------
        frames:
            List of frames ordered in time.
        speed_threshold:
            Pixel distance per frame considered "fast".
        edge_fraction:
            Fraction of the frame (0-1) considered near the edge.

        Returns
        -------
        List of tuples ``(event, frame_index, center)`` where ``event`` is
        either ``"touchdown"`` or ``"exit"``.
        """
        events = []
        last_center = None
        last_index = 0
        for i, frame in enumerate(frames):
            box = self._detect_player(frame)
            if box[0] == -1:
                continue
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if last_center is not None:
                dist = np.hypot(center[0] - last_center[0], center[1] - last_center[1])
                dt = i - last_index
                speed = dist / float(dt) if dt > 0 else 0.0
                h, w = frame.shape[:2]
                near_edge = (
                    center[0] < edge_fraction * w
                    or center[0] > (1.0 - edge_fraction) * w
                    or center[1] < edge_fraction * h
                    or center[1] > (1.0 - edge_fraction) * h
                )
                if speed > speed_threshold and near_edge:
                    events.append(("exit", i, center))
                if speed > speed_threshold and center[1] > (1.0 - edge_fraction) * h:
                    events.append(("touchdown", i, center))
            last_center = center
            last_index = i
        return events

    def classify_video(
        self,
        video_path: str,
        stride: int = 1,
        **kwargs,
    ) -> List[Tuple[str, int, Tuple[int, int]]]:
        """Load a video and classify it for events.

        Parameters
        ----------
        video_path:
            Path to the video file.
        stride:
            Use every ``stride``-th frame for analysis.
        **kwargs:
            Additional keyword arguments forwarded to ``classify_sequence``.
        """
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for video analysis")

        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame = cap.read()
        while success:
            frames.append(frame)
            for _ in range(stride - 1):
                success = cap.grab()
                if not success:
                    break
            success, frame = cap.read()
        cap.release()
        return self.classify_sequence(frames, **kwargs)


__all__ = ["PlayClassifier"]
