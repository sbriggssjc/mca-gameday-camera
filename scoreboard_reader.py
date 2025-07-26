"""Scoreboard state detection using OCR or manual input."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import cv2

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore


@dataclass
class ScoreboardState:
    home: int = 0
    away: int = 0
    quarter: int = 1
    clock: str = "12:00"
    down: Optional[int] = None
    distance: Optional[int] = None


class ScoreboardReader:
    """Detect or manually update scoreboard state."""

    def __init__(self, roi: tuple[int, int, int, int] | None = None) -> None:
        self.roi = roi
        self.state = ScoreboardState()
        self._lock = threading.Lock()
        self._manual_thread = threading.Thread(target=self._manual_input, daemon=True)
        self._manual_thread.start()

    def _manual_input(self) -> None:
        """Background thread for manual scoreboard updates via console."""
        while True:
            try:
                cmd = input(
                    "Enter score as 'home away quarter clock [down distance]' or blank to keep: "
                ).strip()
            except EOFError:
                break
            if not cmd:
                continue
            parts = cmd.split()
            with self._lock:
                try:
                    self.state.home = int(parts[0])
                    self.state.away = int(parts[1])
                    if len(parts) > 2:
                        self.state.quarter = int(parts[2])
                    if len(parts) > 3:
                        self.state.clock = parts[3]
                    if len(parts) > 4:
                        self.state.down = int(parts[4])
                    if len(parts) > 5:
                        self.state.distance = int(parts[5])
                except (IndexError, ValueError):
                    print("Invalid scoreboard input")

    def update(self, frame) -> ScoreboardState:
        """Update state from frame using OCR if possible."""
        if self.roi and pytesseract is not None:
            x, y, w, h = self.roi
            sb = frame[y : y + h, x : x + w]
            gray = cv2.cvtColor(sb, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config="--psm 7")
            digits = [int(s) for s in text.split() if s.isdigit()]
            with self._lock:
                if len(digits) >= 2:
                    self.state.home, self.state.away = digits[:2]
        with self._lock:
            return ScoreboardState(**vars(self.state))
