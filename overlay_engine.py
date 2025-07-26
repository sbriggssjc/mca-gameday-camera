"""Utility for drawing a game overlay on video frames."""

from __future__ import annotations

import cv2

from scoreboard_reader import ScoreboardState


class OverlayEngine:
    """Render scoreboard information on frames."""

    def __init__(self, *, font_scale: float = 1.0, color: tuple[int, int, int] = (255, 255, 255)) -> None:
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.color = color
        self.thickness = 2

    def draw(self, frame, state: ScoreboardState) -> None:
        """Draw the overlay in-place."""
        text = f"Home {state.home} - {state.away} Away  Q{state.quarter}  {state.clock}"
        if state.down is not None and state.distance is not None:
            text += f"  {state.down} & {state.distance}"
        cv2.putText(frame, text, (20, 40), self.font, self.font_scale, self.color, self.thickness, cv2.LINE_AA)
