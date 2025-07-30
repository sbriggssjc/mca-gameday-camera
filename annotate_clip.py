from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2


try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:  # pragma: no cover - depends on OpenCV version
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)


def open_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Open H.264 writer, fallback to MJPG if unavailable."""
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, fps, size)
    return writer


FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_annotation(frame, ann: Dict) -> None:
    """Draw a single annotation on the given frame."""
    color = tuple(int(c) for c in ann.get("color", [0, 0, 255]))
    thickness = int(ann.get("thickness", 2))
    typ = ann.get("type")
    if typ == "circle":
        center = (int(ann["x"]), int(ann["y"]))
        radius = int(ann.get("radius", 20))
        cv2.circle(frame, center, radius, color, thickness)
        label = ann.get("label")
        if label:
            cv2.putText(frame, label, (center[0] + radius + 5, center[1]), FONT, 1.0, color, 2)
    elif typ == "arrow":
        start = tuple(int(v) for v in ann["start"])
        end = tuple(int(v) for v in ann["end"])
        cv2.arrowedLine(frame, start, end, color, thickness)
        label = ann.get("label")
        if label:
            cv2.putText(frame, label, (end[0] + 5, end[1]), FONT, 1.0, color, 2)
    elif typ == "text":
        pos = (int(ann["x"]), int(ann["y"]))
        text = ann.get("text", ann.get("label", ""))
        cv2.putText(frame, text, pos, FONT, 1.0, color, thickness)


def load_annotations(path: str, clip_name: str) -> Dict[int, List[Dict]]:
    """Return mapping of frame number to annotations for the given clip."""
    with open(path) as f:
        data = json.load(f)
    mapping: Dict[int, List[Dict]] = {}
    for item in data:
        if item.get("clip") != clip_name:
            continue
        frame_no = int(item.get("frame", 0))
        mapping.setdefault(frame_no, []).append(item)
    return mapping


def annotate_clip(clip_path: str, annotations_file: str, output_dir: str = "highlight_overlays") -> str:
    clip_p = Path(clip_path)
    anns = load_annotations(annotations_file, clip_p.name)

    cap = cv2.VideoCapture(str(clip_p))
    if not cap.isOpened():
        raise SystemExit(f"Unable to open {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{clip_p.stem}_overlay.mp4"

    writer = open_writer(str(out_path), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit("Failed to open video writer")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        for ann in anns.get(frame_idx, []):
            draw_annotation(frame, ann)
        writer.write(frame)

    cap.release()
    writer.release()
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay annotations on a highlight clip")
    parser.add_argument("clip", help="Input highlight clip (.mp4)")
    parser.add_argument("annotations", help="JSON file with annotations")
    parser.add_argument("--output-dir", default="highlight_overlays", help="Directory for overlay clips")
    args = parser.parse_args()

    out = annotate_clip(args.clip, args.annotations, args.output_dir)
    print(f"Overlay clip saved to {out}")


if __name__ == "__main__":
    main()
