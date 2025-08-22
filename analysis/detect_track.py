"""Player detection and tracking utilities.

The real project would hook up a YOLO/DeepSORT style tracker combined
with an OCR model for jersey recognition.  For the purposes of unit
testing we simply return deterministic pseudo tracking data so later
pipeline stages can operate on predictable structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any

import logging

try:
    import cv2  # type: ignore
    import numpy as np
except Exception:  # pragma: no cover - optional
    cv2 = None  # type: ignore
    np = None  # type: ignore


@dataclass
class Track:
    """Represents a single tracked player instance."""

    frame: int
    player_id: str
    team: str
    jersey_number: str
    bbox: List[int]
    role_hint: str | None = None
    detection_source: str = "primary"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "player_id": self.player_id,
            "team": self.team,
            "jersey_number": self.jersey_number,
            "bbox": self.bbox,
            "role_hint": self.role_hint,
            "detection_source": self.detection_source,
        }


def run(
    video_path: str,
    team: str = "WHITE",
    fps: int = 12,
    model_path: str | None = None,
    settings: Dict[str, Any] | None = None,
) -> List[Track]:
    """Detect and track players in ``video_path``.

    The real project would invoke a heavy detector.  For testing we use a
    lightweight motion-energy approach with an optional motion-blob fallback
    when the primary method fails to detect activity for several frames.
    """

    if model_path:
        print(f"[detect_track] using model: {model_path}")

    if cv2 is None or np is None or not Path(video_path).exists():
        # Generate dummy tracks when video or CV dependencies are missing so
        # downstream code still has data to work with.
        return [
            Track(frame=0, player_id="1", team=team, jersey_number="10", bbox=[0, 0, 10, 10]),
            Track(frame=0, player_id="2", team=team, jersey_number="20", bbox=[20, 0, 30, 10]),
            Track(frame=0, player_id="3", team=team, jersey_number="30", bbox=[40, 0, 50, 10]),
        ]

    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return track_from_frames(frames, team=team, settings=settings)


def _motion_blob_fallback(prev: np.ndarray, cur: np.ndarray, min_area: int) -> tuple[List[List[int]], float]:
    diff = cv2.absdiff(prev, cur)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[List[int]] = []
    max_area = 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append([int(x), int(y), int(x + w), int(y + h)])
            max_area = max(max_area, area)
    return boxes, max_area


def track_from_frames(
    frames: Iterable[np.ndarray],
    *,
    team: str = "WHITE",
    settings: Dict[str, Any] | None = None,
) -> List[Track]:
    """Return tracks from in-memory ``frames`` using motion detection.

    Parameters in ``settings`` control the fallback behaviour:

    - ``enable_motion_blob_fallback`` (bool)
    - ``motion_blob_min_area`` (int)
    - ``motion_blob_confidence`` (float)
    - ``motion_blob_n_frames`` (int)
    """

    if cv2 is None or np is None:
        return []

    settings = settings or {}
    enable_fb = bool(settings.get("enable_motion_blob_fallback", True))
    min_area = int(settings.get("motion_blob_min_area", 150))
    conf_thresh = float(settings.get("motion_blob_confidence", 0.4))
    n_frames = int(settings.get("motion_blob_n_frames", 5))

    logger = logging.getLogger("detect_track")

    tracks: List[Track] = []
    prev_gray: np.ndarray | None = None
    no_motion = 0
    frame_id = 0
    next_id = 1

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev_gray is None:
            prev_gray = blur
            frame_id += 1
            continue

        diff = cv2.absdiff(prev_gray, blur)
        score = float(diff.mean()) / 255.0

        if score >= conf_thresh:
            no_motion = 0
        else:
            no_motion += 1
            if enable_fb and no_motion >= n_frames:
                boxes, max_area = _motion_blob_fallback(prev_gray, blur, min_area)
                if boxes:
                    logger.debug(
                        f"Motion-blob fallback triggered: {len(boxes)} blobs detected, largest area={max_area:.0f}px²"
                    )
                    for box in boxes:
                        tracks.append(
                            Track(
                                frame=frame_id,
                                player_id=f"fb{next_id}",
                                team=team,
                                jersey_number="",
                                bbox=box,
                                detection_source="motion_blob_fallback",
                            )
                        )
                        next_id += 1
                no_motion = 0

        prev_gray = blur
        frame_id += 1

    return tracks


def write_jsonl(tracks: Iterable[Track], path: str) -> None:
    """Write tracks to ``path`` in JSON lines format.

    Each line contains the dictionary representation of a :class:`Track`.
    The function is intentionally straightforward and avoids heavy
    dependencies so it can be easily unit tested.
    """

    import json

    with open(path, "w", encoding="utf8") as f:
        for t in tracks:
            f.write(json.dumps(t.as_dict()) + "\n")
