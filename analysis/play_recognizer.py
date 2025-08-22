"""Play recognition logic."""

from __future__ import annotations

from typing import Dict, Iterable, List


def recognize(plays: Iterable[Dict[str, object]], playbook: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """Match plays against a toy playbook.

    The playbook is a list of dictionaries with keys ``name``, ``formation``
    and optionally ``motion``.  A naive exact match on formation and motion is
    performed.  Recognised plays receive a confidence of ``1.0``; otherwise the
    predicted play is ``"UNKNOWN"`` with confidence ``0.0``.
    """

    predictions: List[Dict[str, object]] = []
    for p in plays:
        formation = p["hash_features"].get("formation")
        motion = p["hash_features"].get("motion")
        match = next(
            (
                pb for pb in playbook
                if pb.get("formation") == formation and pb.get("motion") == motion
            ),
            None,
        )
        if match:
            predictions.append(
                {
                    "play_id": p["play_id"],
                    "predicted_play": match.get("name"),
                    "confidence": 1.0,
                    "formation": formation,
                    "side": match.get("side"),
                    "is_offense_us": True,
                }
            )
        else:
            predictions.append(
                {
                    "play_id": p["play_id"],
                    "predicted_play": "UNKNOWN",
                    "confidence": 0.0,
                    "formation": formation,
                    "side": None,
                    "is_offense_us": True,
                }
            )
    return predictions
