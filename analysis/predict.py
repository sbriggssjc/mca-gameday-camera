from __future__ import annotations

from typing import Any, Callable, Dict, List


def predict_all(
    feats: List[Dict[str, Any]],
    model_predict: Callable[[List[float]], tuple[str, float]],
) -> List[Dict[str, Any]]:
    """Run play predictions for each feature vector.

    Parameters
    ----------
    feats:
        Feature dictionaries produced by :func:`analysis.features.compute_all`.
    model_predict:
        Callable returning ``(label, confidence)`` for a given feature vector.
    """

    rows: List[Dict[str, Any]] = []
    for f in feats:
        sid = f.get("segment_id") or f.get("seg_id")
        feat_dict = f.get("features", {})
        if not feat_dict:
            rows.append({
                "segment_id": sid,
                "play": "UNKNOWN",
                "predicted_play": "UNKNOWN",
                "confidence": 0.0,
                "reason": "no_features",
                "num_players": f.get("num_players", 0),
            })
            continue
        vec = [
            f.get("num_players", 0),
            feat_dict.get("mx", 0.0),
            feat_dict.get("sx", 0.0),
            feat_dict.get("my", 0.0),
            feat_dict.get("sy", 0.0),
            feat_dict.get("spread_x", 0.0),
            feat_dict.get("spread_y", 0.0),
        ]
        label, conf = model_predict(vec)
        rows.append({
            "segment_id": sid,
            "play": label,
            "predicted_play": label,
            "confidence": float(conf),
            "reason": "ok",
            "num_players": f.get("num_players", 0),
        })
    return rows

