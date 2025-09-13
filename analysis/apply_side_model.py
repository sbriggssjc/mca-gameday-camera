from __future__ import annotations
import json, pathlib, numpy as np, sys


def load_model(out: pathlib.Path):
    p = out/"side_model.json"
    if not p.exists():
        raise FileNotFoundError("side_model.json not found")
    return json.loads(p.read_text())


def load_feats(out: pathlib.Path):
    return json.loads((out/"features.json").read_text())


def apply(out_dir: str | pathlib.Path = "output"):
    out = pathlib.Path(out_dir)
    model = load_model(out)
    feat = load_feats(out)

    rows = feat["rows"]
    X = np.array(feat["X"], dtype=float)

    # only centroid model right now
    cents = {k: np.array(v, dtype=float) for k, v in model["centroids"].items()}
    classes = list(cents.keys())

    preds = []
    for i in range(len(rows)):
        dists = {c: float(np.linalg.norm(X[i] - cents[c])) for c in classes}
        # smaller distance = higher confidence
        best = min(dists, key=dists.get)
        # normalize to a [0,1] proxy confidence
        inv = {c: 1.0 / (d + 1e-6) for c, d in dists.items()}
        total = sum(inv.values())
        conf = inv[best] / total if total else 0.5
        preds.append((best, conf))

    # write back into plays.jsonl (lincoln_side_pred / _conf)
    p = out/"plays.jsonl"
    lines = [json.loads(x) if x.strip().startswith("{") else x
             for x in p.read_text().splitlines()]
    j = 0
    new_lines = []
    for line in lines:
        if isinstance(line, dict):
            if j < len(preds):
                side, conf = preds[j]
                line["lincoln_side_pred"] = side
                line["lincoln_side_pred_conf"] = round(conf, 4)
                j += 1
            new_lines.append(json.dumps(line))
        else:
            new_lines.append(line if isinstance(line, str) else json.dumps(line))
    p.write_text("\n".join(new_lines))
    print("[apply_model] wrote model predictions")


if __name__ == "__main__":
    apply(sys.argv[1] if len(sys.argv) > 1 else "output")

