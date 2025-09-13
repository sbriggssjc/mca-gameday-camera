from __future__ import annotations
import json, math, pathlib
from pathlib import Path

FEATURE_KEYS = None

def load_model(out: Path):
    m = json.loads((out/"side_model.json").read_text())
    global FEATURE_KEYS
    FEATURE_KEYS = m["features"]
    return m

def load_feats(out: Path):
    return json.loads((out/"features.json").read_text())

def _vec(d):
    return [float(d.get(k, 0.0)) for k in FEATURE_KEYS]

def _cosine(a,b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) or 1e-6
    nb = math.sqrt(sum(x*x for x in b)) or 1e-6
    return max(0.0, min(1.0, dot/(na*nb)))

def _predict_centroid(feat_vec, centroids):
    # scores are cosine similarity to centroids
    scores = {c: _cosine(feat_vec, v) for c,v in centroids.items()}
    side = max(scores, key=scores.get)
    conf = float(scores[side])
    return side, conf

def apply(out_dir: str):
    out = Path(out_dir)
    model = load_model(out)
    feats = load_feats(out)
    centroids = model["centroids"]

    # update plays.jsonl with side_model_pred fields
    p = out/"plays.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    new_rows = []
    for r in rows:
        if not isinstance(r, dict): new_rows.append(r); continue
        src = r.get("src")
        f = feats.get(src)
        if not f:
            new_rows.append(r); continue
        side, conf = _predict_centroid(_vec(f), centroids)
        r["lincoln_side_model"] = side
        r["lincoln_side_model_conf"] = conf
        new_rows.append(r)

    p.write_text("\n".join(json.dumps(r) for r in new_rows))
    print("[apply_model] wrote model predictions")

if __name__ == "__main__":
    import sys
    apply(sys.argv[1] if len(sys.argv)>1 else "output")
