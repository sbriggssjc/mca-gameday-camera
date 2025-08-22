"""Automatic camera panning using YOLOv8 detections."""

from __future__ import annotations

from typing import Tuple


try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional
    YOLO = None  # type: ignore


class AutoTracker:
    """Crop frames around the main action using YOLOv8."""

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        if YOLO is None:
            raise ImportError("ultralytics package not installed")
        self.model = YOLO(model_path)

    def track(self, frame) -> Tuple[int, int, int, int]:
        """Return crop (x, y, w, h) focused on detected players."""
        height, width = frame.shape[:2]
        res = self.model(frame, verbose=False)[0]
        if not res.boxes:
            return 0, 0, width, height
        boxes = res.boxes.xyxy.cpu().numpy()
        x1 = boxes[:, 0].min()
        y1 = boxes[:, 1].min()
        x2 = boxes[:, 2].max()
        y2 = boxes[:, 3].max()
        # expand box slightly
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int((y2 - y1) * 0.1)
        x1 = max(int(x1 - margin_x), 0)
        y1 = max(int(y1 - margin_y), 0)
        x2 = min(int(x2 + margin_x), width)
        y2 = min(int(y2 + margin_y), height)
        return x1, y1, x2 - x1, y2 - y1
