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
        sufficient = f.get("_sufficient")
        if sufficient is None:
            sufficient = feat_dict.get("_sufficient")
        if sufficient is False:
            rows.append({
                "segment_id": sid,
                "play": "UNKNOWN",
                "predicted_play": "UNKNOWN",
                "confidence": 0.15,
                "reason": "insufficient",
                "num_players": f.get("num_players", 0),
            })
            continue
        vec: List[float] = []
        for key in sorted(feat_dict.keys()):
            val = feat_dict.get(key)
            if isinstance(val, (int, float)):
                vec.append(float(val))
            else:
                vec.append(0.0)
        if not vec:
            rows.append({
                "segment_id": sid,
                "play": "UNKNOWN",
                "predicted_play": "UNKNOWN",
                "confidence": 0.15,
                "reason": "no_features",
                "num_players": f.get("num_players", 0),
            })
            continue
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

