from __future__ import annotations

"""Core data schemas for player identification."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


@dataclass
class PlayerProfile:
    """Visual profile for a player used during identification."""

    player_id: str
    name: Optional[str] = None
    position: Optional[str] = None
    appearance: Dict[str, str] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    notes: Optional[str] = None


@dataclass
class Tracklet:
    """Short term track of a single player within the video."""

    tid: int
    frames: List[int]
    bboxes: List[Tuple[int, int, int, int]]
    embeddings: List[List[float]]
    attributes: Dict[str, object] = field(default_factory=dict)
    assigned_player_id: Optional[str] = None
    confidence: float = 0.0
