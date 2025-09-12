from __future__ import annotations
import json, pathlib, sys

def main(out_dir: str, min_side_conf=0.40):
    out_arg = out_dir
    out = pathlib.Path(out_arg) if out_arg else pathlib.Path("output")

    plays_path = out / "plays.jsonl"
    if not plays_path.exists():
        raise SystemExit(
            f"[reclassify2] plays.jsonl not found at '{plays_path}'. "
            "Tip: verify OUT matches your pipeline run (e.g., OUT=output/opponent_lincoln_20250912) "
            "and call: python -m analysis.reclassify2 \"$OUT\""
        )

    p=plays_path
    rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    seeds={}
    sp=out/"seed_labels.json"
    if sp.exists(): seeds=json.loads(sp.read_text())

    cleaned=[]
    for r in rows:
        src=r.get("src"); phase=r.get("phase","unknown")
        # precedence: manual override > model > heuristic > smoothed
        side = None; conf=0.0
        if src in seeds:
            side=seeds[src]; conf=0.99
        elif r.get("lincoln_side_model"):
            side=r["lincoln_side_model"]; conf=float(r.get("lincoln_side_model_conf",0.5))
        elif r.get("lincoln_side"):
            side=r["lincoln_side"]; conf=float(r.get("lincoln_side_conf",0.3))
        elif r.get("lincoln_side_smoothed"):
            side=r["lincoln_side_smoothed"]; conf=0.35

        # phase gating
        if phase=="special_teams":
            side="unknown"

        r["lincoln_side_final"]=side or "unknown"
        r["lincoln_side_final_conf"]=conf
        r["lincoln_low_conf"] = bool(conf < min_side_conf and r["lincoln_side_final"]!="unknown")
        cleaned.append(r)

    with p.open("w") as w:
        for r in cleaned: w.write(json.dumps(r, ensure_ascii=False)+"\n")
    print("[reclassify2] finalized side with precedence & overrides")

if __name__=="__main__":
    out=sys.argv[1] if len(sys.argv)>1 else ""
    thr=float(sys.argv[2]) if len(sys.argv)>2 else 0.40
    main(out, thr)
