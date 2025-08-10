"""Game analysis orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

# OpenCV is optional. The test environment used in this challenge does not
# include libGL which means ``import cv2`` may fail. We attempt to import it
# but gracefully fall back to ``ffprobe`` via subprocess when unavailable.
try:  # pragma: no cover - exercised indirectly
    import cv2  # type: ignore
except Exception:  # pragma: no cover - import may fail on headless systems
    cv2 = None  # type: ignore
import subprocess

from .models import CoachSummary, PlayAnalysis, PlayerGrade
from .grading import grade_play

# Lightweight player identification pipeline components.  These are simple
# stubs that enable unit tests to exercise the integration points without
# requiring heavy ML dependencies.
from player_id import assign_player_ids, io as pid_io, tracker as pid_tracker
from schemas import Tracklet


def _detect_fps(video_path: str) -> float:
    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        return fps
    # Fallback using ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return 30.0
    if result.returncode != 0:
        return 30.0
    num, denom = result.stdout.strip().split("/")
    try:
        return float(num) / float(denom)
    except Exception:
        return 30.0


def analyze_game(video_path: str, side: str, roster: dict, settings: dict) -> List[PlayAnalysis]:
    """Analyze a game film and return per-play analyses.

    This is a lightweight placeholder that treats the entire video as a
    single play and produces fabricated grades based on the roster.
    """

    fps = _detect_fps(video_path)

    # ------------------------------------------------------------------
    # Player identification: run a stub tracker and attempt to associate
    # tracklets with known player profiles.  The resulting IDs are not used in
    # the toy grading logic but the call verifies the subsystem wiring.
    pid_settings = settings.get("player_id", {})
    player_profiles = pid_io.load_roster(pid_settings.get("roster_path", "data/players.json"))
    tracklets: List[Tracklet] = pid_tracker.track([])
    assign_player_ids(tracklets, player_profiles, pid_settings)

    # In the real system we would split plays and track players. Here we
    # simply fabricate a single play with neutral grades.
    analyses: List[PlayAnalysis] = []
    play = PlayAnalysis(
        play_index=0,
        formation="Unknown",
        play_call="Unknown-O" if side == "offense" else "Base-Run",
        confidence=0.0,
        assignments={},
    )
    for player in roster.keys():
        grade = grade_play(
            play_tracks={},
            call_context={},
            settings=settings,
            player_id=player,
        )
        play.assignments[player] = PlayerGrade(
            player_id=player,
            expected="",
            observed="",
            grade=grade["grade"],
            notes=grade["notes"],
        )
    analyses.append(play)
    return analyses
