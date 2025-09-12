from __future__ import annotations
import json, pathlib, sys


def load(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def save(p, rows):
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(out_dir: str, min_side_conf=0.35, drop_special=True):
    out = pathlib.Path(out_dir)
    p = out / "plays.jsonl"
    rows = load(p)
    cleaned = []
    for r in rows:
        # 1) Drop or mark special teams
        if drop_special and r.get("phase") == "special_teams" and r.get("phase_conf", 0) >= 0.6:
            r["lincoln_side"] = "unknown"  # exclude from offense/defense counts
        # 2) Enforce side confidence
        if r.get("lincoln_side_conf", 0) < min_side_conf:
            r["lincoln_side"] = "unknown"
        # 3) If side is defense but auto run/pass looks like pass with strong horizontal flow, keep as defense (offense of other team)
        # (no change here; included for future refinement)
        cleaned.append(r)
    save(p, cleaned)
    print(f"[reclassify] cleaned: min_side_conf={min_side_conf}, drop_special={drop_special}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "output"
    min_conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
    drop = (sys.argv[3].lower() == "true") if len(sys.argv) > 3 else True
    main(out, min_conf, drop)

