from __future__ import annotations

import json
from pathlib import Path

from .possession import infer_side_by_possession

PRIORITY = [
    "seed",  # from seed_labels.json (absolute)
    "model",  # from side_model.json prediction
    "team_filter",  # initial coarse color/motion filter
    "possession",  # early motion + color dominance
]


def main(out_dir: str, thr: float):
    out = Path(out_dir)
    plays_path = out / "plays.jsonl"
    rows = [
        json.loads(x)
        for x in plays_path.read_text().splitlines()
        if x.strip()
    ]

    seeds: dict[str, str] = {}
    sp = out / "seed_labels.json"
    if sp.exists():
        seeds = json.loads(sp.read_text())

    # read features for possession heuristic support (color ratios)
    feat: dict[str, dict] = {}
    fp = out / "features.json"
    if fp.exists():
        feat = json.loads(fp.read_text())

    new = []
    for r in rows:
        if not isinstance(r, dict):
            new.append(r)
            continue
        src = r.get("src")

        # candidates: (name, side, confidence)
        cands = []

        # 1) seed override
        if src in seeds:
            cands.append(("seed", seeds[src], 0.99))

        # 2) side model prediction
        sm = r.get("lincoln_side_model")
        smc = float(r.get("lincoln_side_model_conf", 0.0))
        if sm:
            cands.append(("model", sm, smc))

        # 3) team_filter baseline
        tf = r.get("lincoln_side")
        tfc = float(r.get("lincoln_side_conf", 0.0))
        if tf:
            cands.append(("team_filter", tf, tfc))

        # 4) possession heuristic
        f = feat.get(src, {})
        br = float(f.get("black_ratio", 0.0))
        wr = float(f.get("white_ratio", 0.0))
        poss_side, poss_conf = infer_side_by_possession(src, br, wr)
        cands.append(("possession", poss_side, poss_conf))

        # choose by PRIORITY, honoring threshold for non-seed
        decided = None
        for name in PRIORITY:
            for nm, sd, cf in cands:
                if nm != name:
                    continue
                if sd == "unknown":
                    continue
                if nm == "seed" or cf >= thr:
                    decided = (sd, cf)
                    break
            if decided:
                break

        if decided is None:
            decided = ("unknown", 0.0)

        r["lincoln_side_final"], r["lincoln_side_final_conf"] = decided
        new.append(r)

    plays_path.write_text("\n".join(json.dumps(r) for r in new))
    print("[reclassify2] finalized side with precedence & overrides")


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "output"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.40
    main(out, thr)

