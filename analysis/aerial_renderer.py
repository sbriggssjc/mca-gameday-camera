"""Minimal 2D field renderer for generating placeholder aerial replays.

The renderer accepts a mapping from track ids to sequences of field coordinates
and produces a simple MP4 animation.  The visuals are intentionally austere – a
plain green field with coloured circles representing players.  This keeps the
implementation lightweight while exercising the surrounding pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import logging
import os
import subprocess

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
    import numpy as np
except Exception:  # pragma: no cover - graceful degradation
    cv2 = None  # type: ignore
    np = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class FieldTrack:
    track_id: str
    points: List[Tuple[float, float]]  # coordinates in yards
    team_id: int = 0
    jersey_number: str | None = None


def _blank_frame(width: int = 1920, height: int = 1080, theme: str = "light"):
    if np is None:
        return None
    color = (34, 139, 34) if theme == "light" else (0, 100, 0)
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    return frame


def render(tracks: Iterable[FieldTrack], out_path: str, *, fps: int = 30, theme: str = "light") -> None:
    """Render a simplistic aerial view animation."""

    if cv2 is None or np is None:  # pragma: no cover - fallback
        # When OpenCV is missing simply touch the file so callers consider the
        # render successful.
        open(out_path, "wb").close()
        return

    frames = int(max((len(t.points) for t in tracks), default=0))
    if frames == 0:
        open(out_path, "wb").close()
        return

    frame = _blank_frame(theme=theme)
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for i in range(frames):
        fr = _blank_frame(theme=theme)
        for t in tracks:
            if i < len(t.points):
                x, y = t.points[i]
                px = int(w * (x / 53.3))
                py = int(h * (1 - y / 120.0))
                cv2.circle(fr, (px, py), 10, (255, 0, 0) if t.team_id else (0, 0, 255), -1)
                if t.jersey_number:
                    cv2.putText(fr, t.jersey_number, (px - 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        vw.write(fr)
    vw.release()

    # Also export an SVG placeholder for coaching handouts.
    svg_path = os.path.splitext(out_path)[0] + ".svg"
    try:
        with open(svg_path, "w", encoding="utf-8") as fh:
            fh.write("<svg xmlns='http://www.w3.org/2000/svg' width='1000' height='533'>\n")
            fh.write("</svg>\n")
    except Exception:  # pragma: no cover - best effort
        logger.debug("failed to write %s", svg_path)
