"""Scoreboard state detection using OCR or manual input."""

from __future__ import annotations

import json
import threading
import time
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
    # Default quarter length is 8:00 minutes per INFC rules
    clock: str = "08:00"
    down: Optional[int] = None
    distance: Optional[int] = None


class ScoreboardReader:
    """Detect or manually update scoreboard state."""

    def __init__(
        self,
        roi: tuple[int, int, int, int] | None = None,
        *,
        ocr_interval: float = 5.0,
        state_file: str = "game_state.json",
    ) -> None:
        self.roi = roi
        self.ocr_interval = ocr_interval
        self.state_file = state_file
        self.state = ScoreboardState()
        self._lock = threading.Lock()
        self._last_ocr = 0.0
        self._manual_thread = threading.Thread(target=self._manual_input, daemon=True)
        self._manual_thread.start()

    def calibrate(self, frame) -> None:
        """Interactively select the scoreboard ROI."""
        self.roi = tuple(int(v) for v in cv2.selectROI("Scoreboard ROI", frame, showCrosshair=False))
        cv2.destroyWindow("Scoreboard ROI")

    def _save_state(self) -> None:
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(vars(self.state), f)
        except Exception:
            pass

    def _filter_digit(self, new: int, prev: int) -> int:
        """Filter common OCR digit mistakes."""
        if new in {6, 8} and prev in {6, 8} and abs(new - prev) == 2:
            return prev
        if new in {1, 7} and prev in {1, 7} and abs(new - prev) == 6:
            return prev
        return new

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
                    self._save_state()
                except (IndexError, ValueError):
                    print("Invalid scoreboard input")

    def update(self, frame) -> ScoreboardState:
        """Update state from frame using OCR if possible."""
        now = time.time()
        if self.roi and pytesseract is not None and now - self._last_ocr >= self.ocr_interval:
            x, y, w, h = self.roi
            sb = frame[y : y + h, x : x + w]
            gray = cv2.cvtColor(sb, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config="--psm 7")
            digits = [int(s) for s in text.split() if s.isdigit()]
            clock_match = None
            colon = text.find(":")
            if colon >= 2:
                candidate = text[colon - 2 : colon + 3]
                if len(candidate) == 5 and candidate[2] == ":":
                    clock_match = candidate
            with self._lock:
                if len(digits) >= 2:
                    self.state.home = self._filter_digit(digits[0], self.state.home)
                    self.state.away = self._filter_digit(digits[1], self.state.away)
                if len(digits) >= 3:
                    self.state.quarter = self._filter_digit(digits[2], self.state.quarter)
                if clock_match and len(clock_match) == 5:
                    self.state.clock = clock_match
                self._save_state()
            self._last_ocr = now
        with self._lock:
            return ScoreboardState(**vars(self.state))
