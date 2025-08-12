"""Simple play classifier with UNKNOWN support."""
from __future__ import annotations

from typing import Dict, List, Any

# constants
MIN_FEATURES = 5          # tune if needed
UNKNOWN_THRESH = 0.60     # <60% conf → UNKNOWN


def to_vector(feats: Any) -> List[float] | None:
    """Convert feature mapping to a dense list."""
    if feats is None:
        return None
    if isinstance(feats, dict):
        return list(feats.values())
    if isinstance(feats, (list, tuple)):
        return list(feats)
    return None


def predict_play_for_segment(feats: Any, model: Any, label_map: Dict[int, str]) -> Dict[str, Any]:
    """
    feats: dict or list of derived features for the window
    returns: {"predicted_play": str, "confidence": float, "reasons": list[str]}
    """
    reasons: List[str] = []
    fv = to_vector(feats)

    if fv is None or len(fv) < MIN_FEATURES:
        reasons.append(f"insufficient_features:{0 if fv is None else len(fv)}")
        return {"predicted_play": "UNKNOWN", "confidence": 0.0, "reasons": reasons}

    proba = model.predict_proba([fv])[0]
    k = int(max(range(len(proba)), key=lambda i: proba[i]))
    conf = float(proba[k])
    label = label_map.get(k, "UNKNOWN")

    if conf < UNKNOWN_THRESH:
        reasons.append(f"low_conf:{conf:.2f}")
        return {"predicted_play": "UNKNOWN", "confidence": conf, "reasons": reasons}

    return {"predicted_play": label, "confidence": conf, "reasons": reasons}
