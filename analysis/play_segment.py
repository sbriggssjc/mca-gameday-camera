"""Play segmentation heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class Play:
    play_id: int
    start_s: float
    end_s: float
    offense_color: str
    defense_color: str
    hash_features: Dict[str, str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "play_id": self.play_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "offense_color": self.offense_color,
            "defense_color": self.defense_color,
            "hash_features": self.hash_features,
        }


def segment(tracks: Iterable[Dict[str, object]]) -> List[Play]:
    """Return a single dummy play spanning from 0 to 5 seconds."""

    return [
        Play(
            play_id=1,
            start_s=0.0,
            end_s=5.0,
            offense_color="WHITE",
            defense_color="DARK",
            hash_features={"formation": "Rit", "motion": "sweep"},
        )
    ]
