from pathlib import Path
from typing import Iterable, Tuple

try:  # pragma: no cover
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .auto_view import SmoothBox, moving_roi, crop_to_box


def write_clip(video_path: str, out_path: str, timecodes: Tuple[int, int], auto_zoom: bool = False):
    """Write a clip between frame indices in ``timecodes``.

    Parameters are purposely simple; this is a resilience-focused stub that
    ensures the VideoWriter is closed even if frame processing fails.
    """
    start_f, end_f = timecodes
    if cv2 is None:  # graceful fallback
        Path(out_path).touch()
        return
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    box_smoother = SmoothBox()
    ret, prev = cap.read()
    frame_idx = start_f
    try:
        while ret and frame_idx <= end_f:
            frame = prev
            if auto_zoom and frame_idx > start_f:
                box = moving_roi(frame, prev)
                box = box_smoother.update(box)
                frame = crop_to_box(frame, box)
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            frame_idx += 1
            ret, prev = cap.read()
    finally:
        writer.release()
        cap.release()
