"""Detection and tracking entry points.

This module wraps a very small YOLOv8 + ByteTrack pipeline.  The goal is to
provide a consistent API for the rest of the analysis stack without pulling in
heavy third‑party dependencies during unit testing.  When the required
libraries are not available (for example in a CI environment) the detector
gracefully falls back to returning an empty track set.

The output format mirrors the contract described in the upgrade notes: a
dictionary with ``fps`` and ``frame_count`` fields and a ``frames`` list
containing per‑frame track dictionaries.  Each track dictionary exposes the
fields ``track_id``, ``x1``, ``y1``, ``x2``, ``y2`` and ``cls``.

Example
-------

>>> run_det_track("clip.mp4")
{'fps': 30.0, 'frame_count': 120, 'frames': [...]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

try:  # pragma: no cover - optional heavy deps
    import cv2  # type: ignore
    from ultralytics import YOLO  # type: ignore
    from .third_party.bytetrack import ByteTracker
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    YOLO = None  # type: ignore
    ByteTracker = None  # type: ignore


def run_det_track(video_path: str) -> Dict[str, object]:
    """Run person detection and ByteTrack tracking on ``video_path``.

    If the required dependencies are missing or the video cannot be read, an
    empty result with ``frame_count`` set to ``0`` is returned.  This keeps
    downstream modules operational in minimal test environments.
    """

    if cv2 is None or YOLO is None or ByteTracker is None:
        return {"fps": 0.0, "frame_count": 0, "frames": []}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"fps": 0.0, "frame_count": 0, "frames": []}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    model = YOLO("yolov8n.pt")
    tracker = ByteTracker()

    frames: List[List[Dict[str, float]]] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections: List[tuple] = []
        try:  # ultralytics API can differ slightly between versions
            results = model(frame, verbose=False)[0]
            boxes = getattr(results, "boxes", None)
            if boxes is not None:
                for xyxy, cls, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
                    if int(cls) == 0:  # person class
                        detections.append((*xyxy, float(conf)))
        except Exception:
            detections = []

        active = tracker.update(detections)
        frame_tracks = [
            {
                "track_id": t["id"],
                "x1": t["bbox"][0],
                "y1": t["bbox"][1],
                "x2": t["bbox"][2],
                "y2": t["bbox"][3],
                "cls": 0,
            }
            for t in active
        ]
        frames.append(frame_tracks)

    cap.release()
    return {"fps": float(fps), "frame_count": frame_count, "frames": frames}


def write_tracks_json(out_dir: Path, clip_path: str, tracks: Dict[str, object]) -> None:
    """Write ``tracks`` to ``OUT/tracks/<clip_basename>.json``."""

    out_dir = Path(out_dir)
    tracks_dir = out_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    base = Path(clip_path).stem
    with (tracks_dir / f"{base}.json").open("w", encoding="utf-8") as f:
        json.dump(tracks, f)


def _main() -> None:  # pragma: no cover - CLI helper
    import argparse

    ap = argparse.ArgumentParser(description="detect and track players")
    ap.add_argument("out_dir")
    ap.add_argument("--clip", required=True)
    args = ap.parse_args()

    tracks = run_det_track(args.clip)
    write_tracks_json(Path(args.out_dir), args.clip, tracks)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    _main()

