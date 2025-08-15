from __future__ import annotations

# Simple per-player grading helpers
from .grader import load_weights, grade_players

# --- Legacy grading system retained for compatibility ---
import os
from typing import Any, Dict, Iterable, List
from ..assignments import Playbook

# ---------------------------------------------------------------------------
# Weight handling
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "EDGE": {
        "contain": 1.0,
        "leverage": 1.0,
        "depth_control": 0.5,
        "pursuit": 0.5,
        "assignment": 1.0,
    },
    "DT": {
        "gap_fill": 1.2,
        "knockback": 1.0,
        "shed": 0.6,
        "assignment": 1.2,
    },
    "LB": {
        "read_first": 1.0,
        "correct_fill_or_drop": 1.0,
        "pursuit_angle": 0.6,
        "assignment": 1.0,
    },
    "DB": {
        "cushion_depth": 1.0,
        "leverage": 0.6,
        "break_on_ball": 0.6,
        "assignment": 0.8,
    },
}

DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "configs", "grading_weights_defense.yaml"
)

POSITION_GROUP = {
    "LE": "EDGE",
    "RE": "EDGE",
    "Monster": "EDGE",
    "Blood": "EDGE",
    "DT1": "DT",
    "DT3": "DT",
    "Mike": "LB",
    "Will": "LB",
    "FS": "DB",
    "LCB": "DB",
    "RCB": "DB",
}

FAIL_SIGNALS = {
    "EDGE": {
        "contain": "lost_edge",
        "leverage": "lost_leverage",
        "depth_control": "depth_violation",
        "pursuit": "pursuit_miss",
        "assignment": "assignment_error",
    },
    "DT": {
        "gap_fill": "wrong_gap",
        "knockback": "no_knockback",
        "shed": "no_shed",
        "assignment": "assignment_error",
    },
    "LB": {
        "read_first": "late_read",
        "correct_fill_or_drop": "wrong_gap",
        "pursuit_angle": "bad_angle",
        "assignment": "assignment_error",
    },
    "DB": {
        "cushion_depth": "depth_violation",
        "leverage": "lost_leverage",
        "break_on_ball": "late_break",
        "assignment": "assignment_error",
    },
}


def _load_weights(path: str | None) -> Dict[str, Dict[str, float]]:
    target = path or DEFAULT_WEIGHTS_PATH
    weights = DEFAULT_WEIGHTS
    if os.path.exists(target):
        try:
            import yaml  # type: ignore

            with open(target, "r", encoding="utf8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                weights = data  # type: ignore[assignment]
        except Exception:
            pass
    return weights


def _grade_player(group: str, signals: Dict[str, Any], weights: Dict[str, Dict[str, float]]):
    metrics = weights.get(group, {})
    total = sum(metrics.values()) or 1.0
    score = 0.0
    mistakes: List[str] = []
    positives: List[str] = []
    for metric, weight in metrics.items():
        failure_key = FAIL_SIGNALS.get(group, {}).get(metric, "")
        failed = signals.get(failure_key, False)
        if failed:
            mistakes.append(metric)
        else:
            positives.append(metric)
            score += weight
    grade = round((score / total) * 4, 2)
    return grade, mistakes, positives


def grade(
    predictions: Iterable[Dict[str, Any]],
    tracks: Iterable[Any],
    identity_map: Dict[str, str],
    playbook: Playbook | None = None,
    weights_path: str | None = None,
) -> List[Dict[str, Any]]:
    """Return grades for each play with defensive evaluation."""

    weights = _load_weights(weights_path)
    results: List[Dict[str, Any]] = []
    for pred in predictions:
        players: Dict[str, Dict[str, Any]] = {}
        for t in tracks:
            pid = identity_map.get(t.player_id, t.player_id)
            role = getattr(t, "role_hint", None)
            group = POSITION_GROUP.get(role, "DB")
            signals = getattr(t, "signals", {}) or {}
            grade_score, mistakes, positives = _grade_player(group, signals, weights)
            players[pid] = {
                "grade": grade_score,
                "notes": [],
                "mistakes": mistakes,
                "positives": positives,
                "position": role,
            }
        results.append(
            {
                "play_id": pred["play_id"],
                "recognized_play": {
                    "name": pred.get("predicted_play"),
                    "confidence": pred.get("confidence", 0.0),
                },
                "players": players,
                "team_highlights": [],
            }
        )
    return results
