"""Player detection and tracking utilities.

The real project would hook up a YOLO/DeepSORT style tracker combined
with an OCR model for jersey recognition.  For the purposes of unit
testing we simply return deterministic pseudo tracking data so later
pipeline stages can operate on predictable structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Any


@dataclass
class Track:
    """Represents a single tracked player instance."""

    frame: int
    player_id: str
    team: str
    jersey_number: str
    bbox: List[int]
    role_hint: str | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "player_id": self.player_id,
            "team": self.team,
            "jersey_number": self.jersey_number,
            "bbox": self.bbox,
            "role_hint": self.role_hint,
        }


def run(
    video_path: str, team: str = "WHITE", fps: int = 12, model_path: str | None = None
) -> List[Track]:
    """Fake detection routine used for tests.

    Parameters
    ----------
    video_path:
        Path to the source video.  The file is **not** opened; the value is
        only persisted in logs so the function works in environments without
        any media files.
    team:
        Team colour for our side.  Tracked players are tagged with this value.
    fps:
        Sampling rate.  Included for API compatibility only.

    Returns
    -------
    list of :class:`Track`
        A minimal set of tracks representing three players.  The bounding
        boxes are arbitrary and only exist to satisfy the tracking schema.
    """

    if model_path:
        print(f"[detect_track] using model: {model_path}")

    # Generate a couple of dummy tracks so downstream modules have something
    # to work with.  In a real implementation these would be derived from
    # model predictions. ``model_path`` is accepted to mirror the real API and
    # allow callers to explicitly provide detector weights.
    tracks = [
        Track(frame=0, player_id="1", team=team, jersey_number="10", bbox=[0, 0, 10, 10]),
        Track(frame=0, player_id="2", team=team, jersey_number="20", bbox=[20, 0, 30, 10]),
        Track(frame=0, player_id="3", team=team, jersey_number="30", bbox=[40, 0, 50, 10]),
    ]
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
