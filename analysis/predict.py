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
        if not f.get("ok"):
            rows.append({
                "segment_id": sid,
                "predicted_play": "UNKNOWN",
                "confidence": 0.0,
                "why": f.get("why", "unknown"),
                "n_players": f.get("n_players", 0),
            })
            continue
        vec = [
            f["n_players"],
            f["mx"],
            f["sx"],
            f["my"],
            f["sy"],
            f["spread_x"],
            f["spread_y"],
        ]
        label, conf = model_predict(vec)
        rows.append({
            "segment_id": sid,
            "predicted_play": label,
            "confidence": float(conf),
            "why": "ok",
            "n_players": f.get("n_players", 0),
        })
    return rows

