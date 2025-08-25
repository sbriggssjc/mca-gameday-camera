"""Rule-aware play classification with fallback heuristics."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class PlayCandidate:
    name: str
    confidence: float


def _motion_hint(cues: Dict[str, Any]) -> str:
    """
    Very light heuristics from pipeline cues you already compute/track.
    Expected keys (if present): 'pre_snap_motion', 'qb_drop_depth', 'handoff_time', 'release_time'
    """
    motion = cues.get("pre_snap_motion")  # e.g., "jet", "orbit", None
    drop = float(cues.get("qb_drop_depth", 0.0))
    handoff = cues.get("handoff_time") is not None
    release = cues.get("release_time") is not None
    if motion == "jet":
        return "JET"
    if handoff and not release:
        return "RUN"
    if release or drop >= 2.5:
        return "PASS"
    return "UNKNOWN"


def _rule_fallback(formation: str, cues: Dict[str, Any]) -> List[PlayCandidate]:
    """
    Map coarse situation to play families to avoid overpredicting F Stick.
    This does NOT override a confident ML call; it only helps when the
    top model score is low or flat.
    """
    form = (formation or "").lower()
    hint = _motion_hint(cues)
    cands: List[PlayCandidate] = []

    # Jet motion strongly biases to Jet Sweep
    if hint == "JET":
        cands.append(PlayCandidate("Reo Jet Sweep" if "reo" in form else ("Leo Jet Sweep" if "leo" in form else "Jet Sweep"), 0.60))
        return cands

    # If it's a run (handoff) from under center/Trips, steer into Dive/Power/Counter family
    if hint == "RUN":
        # Use left/right bias from formation
        if "rit" in form or "right" in form or "trips right" in form:
            cands.extend([PlayCandidate("Rit Dive", 0.40),
                          PlayCandidate("Rit Power R", 0.35),
                          PlayCandidate("Rit F Counter", 0.30)])
        elif "lit" in form or "left" in form or "trips left" in form:
            cands.extend([PlayCandidate("Lit Dive", 0.40),
                          PlayCandidate("Lit Power L", 0.35),
                          PlayCandidate("Lit F Counter", 0.30)])
        else:
            cands.extend([PlayCandidate("Dive", 0.35), PlayCandidate("Power", 0.30), PlayCandidate("F Counter", 0.25)])
        return cands

    # If clear pass/dropback: restrict Stick predictions to Reo/Leo families
    if hint == "PASS":
        if "reo" in form:
            cands.extend([PlayCandidate("Reo F Stick", 0.55), PlayCandidate("Reo Flood", 0.45), PlayCandidate("Reo Quick Screen", 0.35)])
        elif "leo" in form:
            cands.extend([PlayCandidate("Leo F Stick", 0.55), PlayCandidate("Leo Flood", 0.45), PlayCandidate("Leo Quick Screen", 0.35)])
        else:
            # In Trips w/o Reo/Leo tags, be more agnostic
            if "trips right" in form:
                cands.extend([PlayCandidate("Flood", 0.45), PlayCandidate("Quick Screen", 0.40)])
            elif "trips left" in form:
                cands.extend([PlayCandidate("Flood", 0.45), PlayCandidate("Quick Screen", 0.40)])
            else:
                cands.extend([PlayCandidate("Quick Screen", 0.40), PlayCandidate("Flood", 0.35)])
        return cands

    # Unknown → no strong suggestion
    return []


def classify_play(frame_window, playbook, formation, logger, cues=None):
    if cues is None:
        cues = {}
    # === 1) model inference (existing) ===
    # Replace the placeholder below with your actual model scoring
    model_candidates: List[PlayCandidate] = []
    try:
        # Example: fill model_candidates with (name, score) from your current classifier
        pass
    except Exception as e:
        logger.warning(f"[play_classifier] model error -> using fallback: {e}")

    # === 2) If model is indecisive/flat, use rule fallback ===
    def _best(cands: List[PlayCandidate]) -> PlayCandidate:
        return max(cands, key=lambda c: c.confidence)

    if not model_candidates or (len(model_candidates) >= 2 and abs(model_candidates[0].confidence - model_candidates[1].confidence) < 0.05):
        fb = _rule_fallback(formation, cues)
        if fb:
            best = _best(fb)
            logger.info(f"[play_classifier] FALLBACK -> {best.name} conf={best.confidence:.2f}")
            return {"name": best.name, "confidence": best.confidence, "candidates": [c.__dict__ for c in fb]}
        # last resort
        return {"name": "Unknown", "confidence": 0.20, "candidates": []}

    # === 3) Normal confident return ===
    best = _best(model_candidates)
    others = [c for c in model_candidates if c.name != best.name][:4]
    return {"name": best.name, "confidence": best.confidence, "candidates": [c.__dict__ for c in others]}


__all__ = ["classify_play", "PlayCandidate"]

