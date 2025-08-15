from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math
from ..playbook.schema import PlaybookIndex


def _cosine(a: List[float], b: List[float]) -> float:
    num = sum(x*y for x,y in zip(a,b))
    da = math.sqrt(sum(x*x for x in a)) + 1e-6
    db = math.sqrt(sum(y*y for y in b)) + 1e-6
    return num / (da*db)


def formation_features(presnap: Dict[str, Any]) -> Dict[str, Any]:
    """Build normalized vector: width splits, stack counts, backfield depth bins, strength."""
    # Inputs expected from your detector/tracking: presnap["players"] with {role,x_norm,y_bins}
    # Be tolerant: if roles missing, use clusters by x position.
    vec = presnap.get("vector", [])
    meta = {
        "personnel": presnap.get("personnel"),     # if OCR/ID available
        "strength": presnap.get("strength"),       # Left/Right/Boundary/Field
    }
    return {"vec": vec, "meta": meta}


def match_formation(pb: PlaybookIndex, presnap: Dict[str, Any], topk: int = 3) -> List[Tuple[str, float]]:
    q = formation_features(presnap)
    qv = q["vec"] or [0]*12
    scores: List[Tuple[str, float]] = []
    for name, fs in pb.formations.items():
        fv = (fs.anchors or {}).get("vec", [0]*12)
        s = _cosine(qv, fv)
        # bonus for personnel or strength tag matches
        if q["meta"].get("personnel") and fs.personnel == q["meta"]["personnel"]:
            s += 0.05
        if q["meta"].get("strength") and fs.side and fs.side == q["meta"]["strength"]:
            s += 0.03
        scores.append((name, float(max(-1.0, min(1.0, s)))))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topk]
