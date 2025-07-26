# pyright: reportMissingImports=false
"""YOLOv8-based auto cropper with object tracking.

This module detects players each frame with YOLOv8 and keeps tracks
across frames using either the ``deep_sort_realtime`` package if
available or OpenCV's ``MultiTracker`` as a fallback.  The crop region
is derived from the union of active tracks and expanded slightly to
provide some margin around the action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional
    YOLO = None  # type: ignore

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:  # pragma: no cover - optional
    DeepSort = None  # type: ignore

try:
    import cv2
except Exception:  # pragma: no cover - optional
    cv2 = None  # type: ignore


@dataclass
class TrackBox:
    """Simple bounding box for a track."""

    x1: int
    y1: int
    x2: int
    y2: int

    def update(self, box: Tuple[int, int, int, int]) -> None:
        self.x1, self.y1, self.x2, self.y2 = box

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class SmartAutoTracker:
    """Track player clusters and return crop focusing on center of activity."""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        if YOLO is None:
            raise ImportError("ultralytics package not installed")
        if cv2 is None:
            raise ImportError("opencv-python package not installed")

        self.model = YOLO(model_path)
        if DeepSort is not None:
            self.tracker = DeepSort(max_age=15)
            self.cv_trackers = None
        else:
            self.tracker = None
            self.cv_trackers = cv2.legacy.MultiTracker_create()
        self.track_boxes: dict[int, TrackBox] = {}
        self.next_id = 0

    def _init_cv_tracks(self, detections: List[Tuple[int, int, int, int]], frame: np.ndarray) -> None:
        """Initialize OpenCV trackers for the given detections."""
        self.cv_trackers = cv2.legacy.MultiTracker_create()
        self.track_boxes.clear()
        for box in detections:
            x1, y1, x2, y2 = box
            self.track_boxes[self.next_id] = TrackBox(x1, y1, x2, y2)
            tracker = cv2.legacy.TrackerCSRT_create()
            self.cv_trackers.add(tracker, frame, (x1, y1, x2 - x1, y2 - y1))
            self.next_id += 1

    def _update_cv_tracks(self, frame: np.ndarray) -> None:
        """Update OpenCV trackers."""
        if self.cv_trackers is None:
            return
        success, boxes = self.cv_trackers.update(frame)
        if not success:
            return
        for idx, (x, y, w, h) in enumerate(boxes):
            track = self.track_boxes.get(idx)
            if track:
                track.update((int(x), int(y), int(x + w), int(y + h)))

    def _detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        res = self.model(frame, verbose=False)[0]
        boxes = []
        for xyxy in res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy[:4])
            boxes.append((x1, y1, x2, y2))
        return boxes

    def _tracks_union(self) -> Tuple[int, int, int, int] | None:
        boxes = [b.as_tuple() for b in self.track_boxes.values()]
        if not boxes:
            return None
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return x1, y1, x2, y2

    def track(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Return crop (x, y, w, h) centered on detected player cluster."""
        height, width = frame.shape[:2]
        detections = self._detect(frame)
        if self.tracker is not None:
            tracks = self.tracker.update_tracks(detections, frame=frame)
            self.track_boxes.clear()
            for t in tracks:
                if not t.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, t.to_ltrb())
                self.track_boxes[t.track_id] = TrackBox(x1, y1, x2, y2)
        else:
            if not self.track_boxes:
                self._init_cv_tracks(detections, frame)
            else:
                self._update_cv_tracks(frame)

        union = self._tracks_union()
        if union is None:
            return 0, 0, width, height

        x1, y1, x2, y2 = union
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int((y2 - y1) * 0.1)
        x1 = max(x1 - margin_x, 0)
        y1 = max(y1 - margin_y, 0)
        x2 = min(x2 + margin_x, width)
        y2 = min(y2 + margin_y, height)
        return x1, y1, x2 - x1, y2 - y1
