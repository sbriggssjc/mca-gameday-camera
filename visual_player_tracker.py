"""Pose-based player tracker for highlight clips.

This module detects players using YOLOv8 pose models, matches jersey
numbers via OCR and computes simple motion metrics for each player in a
video clip. Results are written per player to ``visual_metrics``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("ultralytics is required") from exc

try:
    import easyocr
except Exception:
    easyocr = None  # type: ignore

import roster


@dataclass
class PlayerState:
    """Internal state for one player during a clip."""

    jersey: int
    prev_center: Optional[Tuple[float, float]] = None
    last_angle: Optional[float] = None
    distance: float = 0.0
    max_speed: float = 0.0
    direction_changes: int = 0
    contact_events: int = 0
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)


class VisualPlayerTracker:
    """Track players using pose detection and simple heuristics."""

    def __init__(self, model_path: str = "yolov8n-pose.pt") -> None:
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(["en"], gpu=False) if easyocr else None

    def _ocr_number(self, frame: np.ndarray, box: np.ndarray) -> Optional[int]:
        if self.reader is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        result = self.reader.readtext(crop, detail=0)
        for text in result:
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    continue
        return None

    def _update_state(
        self,
        state: PlayerState,
        center: Tuple[float, float],
        dt: float,
        t: float,
    ) -> None:
        if state.prev_center is not None:
            dx = center[0] - state.prev_center[0]
            dy = center[1] - state.prev_center[1]
            dist = math.hypot(dx, dy)
            state.distance += dist
            speed = dist / dt
            state.max_speed = max(state.max_speed, speed)
            angle = math.atan2(dy, dx)
            if state.last_angle is not None:
                delta = abs(angle - state.last_angle)
                if delta > math.pi:
                    delta = 2 * math.pi - delta
                if delta > math.radians(45):
                    state.direction_changes += 1
            state.last_angle = angle
        state.prev_center = center
        state.trajectory.append((t, center[0], center[1]))

    def _update_contacts(
        self,
        states: Dict[int, PlayerState],
        last_dists: Dict[Tuple[int, int], float],
    ) -> None:
        jerseys = sorted(states.keys())
        for i in range(len(jerseys)):
            for j in range(i + 1, len(jerseys)):
                a_id, b_id = jerseys[i], jerseys[j]
                a_center = states[a_id].prev_center
                b_center = states[b_id].prev_center
                if a_center is None or b_center is None:
                    continue
                d = math.hypot(a_center[0] - b_center[0], a_center[1] - b_center[1])
                key = (a_id, b_id)
                last = last_dists.get(key)
                if last is not None and last - d > 10 and d < 40:
                    states[a_id].contact_events += 1
                    states[b_id].contact_events += 1
                last_dists[key] = d

    def _finalize_state(self, state: PlayerState) -> Dict[str, object]:
        return {
            "jersey": state.jersey,
            "name": roster.ROSTER.get(state.jersey, {}).get("name"),
            "distance": state.distance,
            "max_speed": state.max_speed,
            "direction_changes": state.direction_changes,
            "contact_events": state.contact_events,
        }

    def track_clip(self, clip_path: str) -> Dict[int, Dict[str, object]]:
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {clip_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dt = 1.0 / fps
        states: Dict[int, PlayerState] = {}
        last_dists: Dict[Tuple[int, int], float] = {}
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            t = frame_id * dt
            res = self.model(frame, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else np.empty((0, 4))
            kps = res.keypoints.xyn.cpu().numpy() if res.keypoints else None
            for i, box in enumerate(boxes):
                jersey = self._ocr_number(frame, box)
                if jersey is None or jersey not in roster.ROSTER:
                    continue
                if kps is not None and i < kps.shape[0]:
                    kp = kps[i]
                    if kp.shape[0] > 12:
                        lh = kp[11]
                        rh = kp[12]
                        center = ((lh[0] + rh[0]) / 2 * frame.shape[1], (lh[1] + rh[1]) / 2 * frame.shape[0])
                    else:
                        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                else:
                    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                state = states.setdefault(jersey, PlayerState(jersey))
                self._update_state(state, center, dt, t)
            self._update_contacts(states, last_dists)
        cap.release()
        return {j: self._finalize_state(s) for j, s in states.items()}

    def save_metrics(self, metrics: Dict[int, Dict[str, object]], clip_name: str) -> None:
        out_dir = Path("visual_metrics")
        out_dir.mkdir(exist_ok=True)
        for jersey, data in metrics.items():
            out_path = out_dir / f"player_{jersey}_{clip_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)


def analyze_clip(clip_path: str) -> None:
    tracker = VisualPlayerTracker()
    clip_name = Path(clip_path).stem
    metrics = tracker.track_clip(clip_path)
    tracker.save_metrics(metrics, clip_name)
    print(f"Metrics saved for clip {clip_name}")


__all__ = ["analyze_clip", "VisualPlayerTracker"]
