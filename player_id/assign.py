"""Assignment of tracklets to known player profiles."""

from __future__ import annotations

from typing import Dict, List
import numpy as np

from schemas import PlayerProfile, Tracklet
from .reid import tracklet_signature


def _color_distance(attrs: Dict[str, str], appearance: Dict[str, str]) -> float:
    """Simple attribute distance: count mismatched string values."""

    dist = 0.0
    for key, val in appearance.items():
        if attrs.get(key) != val:
            dist += 1.0
    return dist


def assign_player_ids(
    tracklets: List[Tracklet], roster: List[PlayerProfile], rules: Dict
) -> List[Tracklet]:
    """Assign stable player IDs to tracklets.

    The matching uses a simple weighted distance combining cosine distance of
    embeddings and mismatches in appearance attributes.  It greedily assigns
    the best matching profile to each tracklet and records a confidence score.
    """

    profiles = {p.player_id: p for p in roster}
    w_emb = rules.get("w_emb", 0.65)
    w_attr = rules.get("w_attr", 0.30)
    threshold = rules.get("confidence_threshold", 0.0)

    for t in tracklets:
        sig = tracklet_signature(t)
        best_pid = None
        best_score = float("inf")
        for pid, prof in profiles.items():
            emb = np.array(prof.embedding or np.zeros_like(sig["avg_emb"]))
            if emb.size == 0:
                d_emb = 1.0
            else:
                d_emb = 1.0 - float(np.dot(sig["avg_emb"], emb))
            d_attr = _color_distance(sig["attr"], prof.appearance)
            score = w_emb * d_emb + w_attr * d_attr
            if score < best_score:
                best_score = score
                best_pid = pid
        confidence = 1.0 - best_score
        if confidence < threshold:
            t.assigned_player_id = None
        else:
            t.assigned_player_id = best_pid
        t.confidence = confidence
    return tracklets
