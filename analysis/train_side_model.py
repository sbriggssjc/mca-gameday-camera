from __future__ import annotations
import json, pathlib, statistics
from pathlib import Path

# very small centroid classifier to avoid sklearn dependency
SIDE_INDEX = {"offense":0, "defense":1, "special_teams":2}

FEATURE_KEYS = [
    "black_ratio","white_ratio",
    "flow_mean","flow_std","vx_mean","vy_mean",
    "mb_minus_mw_mean"
]

def _load_feats(out: Path):
    p = out/"features.json"
    feat = json.loads(p.read_text())
    return {k: v for k, v in feat.items()}

def _load_seeds(out: Path):
    p = out/"seed_labels.json"
    if not p.exists(): return {}
    return json.loads(p.read_text())

def _vec(d):
    return [float(d.get(k, 0.0)) for k in FEATURE_KEYS]

def _centroid_train(feats: dict, seeds: dict):
    # collect per-class vectors from seeds
    byc = {k: [] for k in SIDE_INDEX}
    for src, side in seeds.items():
        f = feats.get(src)
        if not f: continue
        byc[side].append(_vec(f))
    # need at least two classes to be useful
    classes = [c for c, arr in byc.items() if len(arr) >= 2]
    if len(classes) < 2:
        raise RuntimeError("need >=2 classes with >=2 seeds each")
    centroids = {}
    for c, arr in byc.items():
        if arr:
            cols = list(zip(*arr))
            centroids[c] = [statistics.fmean(col) for col in cols]
    return {
        "type":"centroid",
        "features": FEATURE_KEYS,
        "centroids": centroids
    }

def main(out_dir: str):
    out = Path(out_dir)
    feats = _load_feats(out)
    seeds = _load_seeds(out)
    if not seeds:
        raise RuntimeError("no seed_labels.json; seed a few obvious clips first")

    model = _centroid_train(feats, seeds)
    (out/"side_model.json").write_text(json.dumps(model))
    print("[train] wrote", out/"side_model.json")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "output")
