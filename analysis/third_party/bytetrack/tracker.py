"""Very small ByteTrack style tracker.

The implementation below is intentionally compact.  It performs greedy IOU
matching between existing tracks and new detections and applies a simple
velocity based smoothing.  Only the features that are required by the unit
tests are implemented.  Tracks are represented as dictionaries with the keys
`id` and `bbox` (in ``xyxy`` format).

Example
-------

>>> tracker = ByteTracker()
>>> detections = [(0, 0, 10, 10, 0.9)]  # x1, y1, x2, y2, score
>>> tracker.update(detections)
[{'id': 1, 'bbox': [0, 0, 10, 10]}]

The real ByteTrack algorithm contains many more optimisations and heuristics.
This version is sufficient for tests and for building higher level modules in
the repository without pulling in a heavy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


def _iou(box_a: Tuple[float, float, float, float],
         box_b: Tuple[float, float, float, float]) -> float:
    """Compute intersection over union of two ``xyxy`` boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class _Track:
    bbox: Tuple[float, float, float, float]
    id: int
    vx: float = 0.0
    vy: float = 0.0
    age: int = 0


class ByteTracker:
    """Minimal ByteTrack style tracker.

    Parameters
    ----------
    iou_threshold:
        Minimum IOU required to associate a detection with an existing track.
    max_age:
        Number of consecutive frames a track may remain unmatched before it is
        removed.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self.tracks: List[_Track] = []

    def update(self, detections: List[Tuple[float, float, float, float, float]]):
        """Update tracker with ``detections``.

        Each detection is a tuple ``(x1, y1, x2, y2, score)``.  The score is not
        used in this lightweight implementation but is accepted for API
        compatibility.  Returns a list of active tracks represented as
        dictionaries ``{"id": int, "bbox": [x1, y1, x2, y2]}``.
        """

        assigned = set()
        # Greedy matching
        for track in list(self.tracks):
            best_iou = 0.0
            best_det = None
            for i, det in enumerate(detections):
                if i in assigned:
                    continue
                iou = _iou(track.bbox, det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_det = i
            if best_iou >= self.iou_threshold and best_det is not None:
                dx = detections[best_det][0] - track.bbox[0]
                dy = detections[best_det][1] - track.bbox[1]
                track.vx = 0.8 * track.vx + 0.2 * dx
                track.vy = 0.8 * track.vy + 0.2 * dy
                track.bbox = detections[best_det][:4]
                track.age = 0
                assigned.add(best_det)
            else:
                # decay age and predict simple motion
                track.age += 1
                if track.age > self.max_age:
                    self.tracks.remove(track)
                    continue
                x1, y1, x2, y2 = track.bbox
                track.bbox = (
                    x1 + track.vx,
                    y1 + track.vy,
                    x2 + track.vx,
                    y2 + track.vy,
                )

        # Create new tracks
        for i, det in enumerate(detections):
            if i in assigned:
                continue
            bbox = det[:4]
            self.tracks.append(_Track(bbox=bbox, id=self._next_id))
            self._next_id += 1

        return [
            {"id": t.id, "bbox": [float(x) for x in t.bbox]}
            for t in self.tracks
            if t.age <= self.max_age
        ]

