from __future__ import annotations
import json, pathlib, numpy as np, sys

def load_model(out):
    m=json.loads((out/"side_model.json").read_text())
    return m

def load_feats(out):
    return json.loads((out/"features.json").read_text())

def predict_one(f, M):
    if not f or not f.get("ok"):
        return "unknown", 0.25
    x=np.array([float(f.get(k,0.0)) for k in M["feats"]], dtype=np.float32)
    mu=np.array(M["mu"]); sigma=np.array(M["sigma"])
    xn=(x-mu)/sigma
    W=np.array(M["W"]); b=np.array(M["b"])
    z = xn.dot(W)+b
    z = z - z.max()
    e=np.exp(z); p=e/np.sum(e)
    idx=int(np.argmax(p)); return M["classes"][idx], float(p[idx])

def apply(out_dir: pathlib.Path):
    out = pathlib.Path(out_dir)
    if not (out/"side_model.json").exists():
        print("[apply_model] no side_model.json; skipping apply")
        return
    model=load_model(out); feat=load_feats(out)
    p=out/"plays.jsonl"; rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    upd=[]
    for r in rows:
        src=r.get("src"); f=feat.get(src)
        lab, conf = predict_one(f, model)
        # precedence: manual override (seed_labels.json) will be picked up later by reclassifier
        r["lincoln_side_model"]=lab; r["lincoln_side_model_conf"]=conf
        upd.append(r)
    with p.open("w") as w:
        for r in upd: w.write(json.dumps(r, ensure_ascii=False)+"\n")
    print("[apply_model] wrote model predictions")

def main():
    out_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    out = pathlib.Path(out_arg) if out_arg else pathlib.Path("output")

    plays_path = out / "plays.jsonl"
    if not plays_path.exists():
        raise SystemExit(
            f"[apply_model] plays.jsonl not found at '{plays_path}'. "
            "Tip: verify OUT matches your pipeline run (e.g., OUT=output/opponent_lincoln_20250912) "
            "and call: python -m analysis.apply_side_model \"$OUT\""
        )

    apply(out)

if __name__=="__main__":
    main()
