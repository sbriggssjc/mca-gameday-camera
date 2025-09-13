from __future__ import annotations
import json, pathlib, numpy as np, sys


def load_feats(out: pathlib.Path):
    return json.loads((out/"features.json").read_text())


def load_seeds(out: pathlib.Path):
    p = out/"seed_labels.json"
    return {} if not p.exists() else json.loads(p.read_text())


def save_model(out: pathlib.Path, model):
    (out/"side_model.json").write_text(json.dumps(model))


def train(out_dir: str | pathlib.Path = "output"):
    out = pathlib.Path(out_dir)
    feat = load_feats(out)
    seeds = load_seeds(out)

    # need seeds across >=2 classes
    labels = set(seeds.values())
    if len(seeds) < 3 or len(labels) < 2:
        print("[train] need >=3 total seed examples across >=2 classes")
        return None

    # very small/shallow model: per-class centroids in feature space
    X = []
    y = []
    src_to_idx = {row["src"]: i for i, row in enumerate(feat["rows"])}
    for src, lab in seeds.items():
        idx = src_to_idx.get(src)
        if idx is None:
            continue
        X.append(feat["X"][idx])
        y.append(lab)
    if len(set(y)) < 2:
        print("[train] insufficient class variety after mapping seeds")
        return None

    X = np.array(X, dtype=float)
    y = np.array(y)
    classes = sorted(set(y))
    centroids = {}
    for c in classes:
        centroids[c] = np.mean(X[y == c], axis=0).tolist()

    model = {"type": "centroid", "classes": classes, "centroids": centroids}
    save_model(out, model)
    print(f"[train] wrote {out/'side_model.json'}")
    return model


# Keep old zero-arg entrypoint but make it call the new API
def main(out_dir: str = "output"):
    return train(out_dir)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(out)

