from __future__ import annotations

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

import json
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("ultralytics is required") from exc

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("deep_sort_realtime is required") from exc

try:
    import easyocr
except Exception:
    easyocr = None  # type: ignore


MODEL_PATH = Path("models/yolov8m.pt")
OUTPUT_DIR = Path("outputs")
OUTPUT_VIDEO = OUTPUT_DIR / "annotated_video.mp4"
OUTPUT_JSON = OUTPUT_DIR / "play_metadata.json"


class Tracker:
    """Wrapper around YOLOv8 model and DeepSORT tracker."""

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(MODEL_PATH)
        self.model = YOLO(str(MODEL_PATH))
        self.tracker = DeepSort(max_age=30)
        self.ocr_reader = easyocr.Reader(["en"], gpu=False) if easyocr else None

    def _detect_players(self, frame: np.ndarray) -> List[List[float]]:
        """Return detected bounding boxes as [x1, y1, x2, y2]."""
        result = self.model(frame, verbose=False)[0]
        boxes = []
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box[:4]
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
        return boxes

    def _ocr_number(self, frame: np.ndarray, bbox: List[float]) -> str | None:
        if self.ocr_reader is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        result = self.ocr_reader.readtext(crop, detail=0)
        for text in result:
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                return digits
        return None

    def process(self, cap: cv2.VideoCapture, writer: cv2.VideoWriter) -> List[Dict[str, object]]:
        metadata: List[Dict[str, object]] = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            boxes = self._detect_players(frame)
            tracks = self.tracker.update_tracks(boxes, frame=frame)
            track_data = []
            for trk in tracks:
                if not trk.is_confirmed():
                    continue
                x1, y1, x2, y2 = trk.to_ltrb()
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                jersey = self._ocr_number(frame, bbox)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                label = f"ID {trk.track_id}"
                if jersey:
                    label += f" #{jersey}"
                cv2.putText(
                    frame,
                    label,
                    (bbox[0], max(0, bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                track_data.append({"id": trk.track_id, "bbox": bbox, "jersey": jersey})
            writer.write(frame)
            metadata.append({"frame": frame_id, "time": frame_id / fps, "tracks": track_data})
        return metadata


def analyze_video(video_path: str) -> None:
    """Analyze a video and save annotated output and metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))
    tracker = Tracker()
    metadata = tracker.process(cap, writer)
    cap.release()
    writer.release()
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Annotated video saved to {OUTPUT_VIDEO}")
    logging.info(f"Metadata saved to {OUTPUT_JSON}")
__all__ = ["analyze_video"]
