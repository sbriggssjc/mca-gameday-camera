"""Data models for MCA film analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PlayerTrack:
    """Simple representation of a player's trajectory.

    Attributes:
        player_id: Identifier from the roster.
        positions: List of (x, y) tuples in field coordinates.
    """

    player_id: str
    positions: List[tuple]


@dataclass
class AssignmentExpectation:
    """Expected responsibility for a player on a play."""

    player_id: str
    description: str


@dataclass
class PlayerGrade:
    """Grade awarded to a player for a play."""

    player_id: str
    expected: str
    observed: str
    grade: float
    notes: str = ""


@dataclass
class PlayAnalysis:
    """Full analysis for a single play."""

    play_index: int
    formation: str
    play_call: str
    confidence: float
    assignments: Dict[str, PlayerGrade] = field(default_factory=dict)


@dataclass
class CoachSummary:
    """Aggregated summary for coaches."""

    player_averages: Dict[str, float]
    top_corrections: Dict[str, List[str]]
