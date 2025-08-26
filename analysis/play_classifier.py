from __future__ import annotations
import math, random

MIN_PLAYCALL_CONF = float(os.environ.get("MIN_PLAYCALL_CONF", "0.5")) if "os" in globals() else 0.5
try:
    import os
except:
    import os

def _softmax(scores, temperature: float = 1.0):
    exps = [math.exp(s / max(1e-6, temperature)) for s in scores]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

def _topk(candidates, k=3):
    return sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

def classify_plays(segments, playbook, video_profile=None):
    """
    Return a list of dicts:
      play_id, t0, t1, snap, whistle, clip_path, formation, formation_confidence,
      play_family, playcall_confidence, outcome, clip_duration, candidates=[(name,prob),...]
    """
    plays_def = playbook.get("plays", [])
    known_names = [p.get("name", "") for p in plays_def]

    results = []
    for i, seg in enumerate(segments, start=1):
        # Dummy features -> demo probabilities (replace with real model as available)
        base_scores = [random.random() for _ in known_names]
        probs = _softmax(base_scores, temperature=0.8)
        candidates = list(zip(known_names, probs))
        top = _topk(candidates, 3)

        top_name, top_prob = (top[0] if top else ("Unknown", 0.0))

        # Unknown fallback
        play_family = top_name if top_prob >= MIN_PLAYCALL_CONF else "Unknown"
        playcall_conf = float(top_prob if play_family != "Unknown" else 0.0)

        results.append({
            "play_id": f"PLAY_{i:03d}",
            "t0": seg.get("t0", 0.0),
            "t1": seg.get("t1", 0.0),
            "snap": seg.get("snap", None),
            "whistle": seg.get("whistle", None),
            "clip_path": seg.get("clip_path", ""),
            "formation": seg.get("formation", "Unknown"),
            "formation_confidence": seg.get("formation_confidence", 0.0),
            "play_family": play_family,
            "playcall_confidence": playcall_conf,
            "outcome": seg.get("outcome", ""),
            "clip_duration": round(max(0.0, seg.get("t1", 0.0) - seg.get("t0", 0.0)), 3),
            "candidates": [(n, round(float(p), 3)) for (n, p) in top],
        })
    return results

