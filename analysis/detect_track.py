from __future__ import annotations
from typing import Any, Dict, List

def run(video_path: str, team: str = "", fps: int | None = None, **kwargs: Any) -> Dict[str, Any]:
    """Stub tracker: returns an empty structure but preserves interfaces.
    Replace with your real detector when ready.
    """
    return {
        "team": team,
        "fps": fps,
        "tracks": [],         # list of per-player tracks when you implement
        "detections": [],     # raw detections if you implement
        "meta": {"source": video_path},
    }
