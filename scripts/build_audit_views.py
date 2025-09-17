#!/usr/bin/env python3
from pathlib import Path
import sys, json, csv, collections

ST_EXCLUDE = {"xp","kickoff","punt","return","kneel","spike"}  # match analytics

def norm(v, allowed, default="unknown"):
    v = (v or "").strip().lower()
    return v if v in allowed else default

def rp_from_flags(p):
    r = bool(p.get("is_run")); s = bool(p.get("is_pass"))
    if r and not s: return "run"
    if s and not r: return "pass"
    return "unknown"

def rp_used(p):
    # allow an audit or fixer override if present
    v = (p.get("rp_fix") or p.get("rp") or "").strip().lower()
    if v in {"run","pass","unknown"}:
        return v
    return rp_from_flags(p)

def dir_used(p, rp):
    if rp == "run":
        return norm(p.get("run_dir") or p.get("direction"), {"left","right","unknown"})
    if rp == "pass":
        return norm(p.get("direction"), {"left","right","unknown"})
    return "unknown"

def keep_for_analytics(p, reasons_out):
    # explicit exclude flag
    if p.get("exclude_from_analytics"):
        reasons_out.append("exclude_flag")
        return False
    # special teams labels
    st = (p.get("st_fix") or p.get("st") or "").strip().lower()
    if st in ST_EXCLUDE:
        reasons_out.append(f"st:{st}")
        return False
    # special_teams boolean
    if p.get("special_teams"):
        reasons_out.append("special_teams")
        return False
    # safe phase filter
    ph = (p.get("phase") or "").strip().lower()
    if ph and ph not in {"live","play"}:
        reasons_out.append(f"phase:{ph}")
        return False
    # side must be o/d
    side = (p.get("side") or "").strip().lower()
    if side not in {"offense","defense"}:
        reasons_out.append(f"side:{side or 'none'}")
        return False
    return True

def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/opponent_jenks_silver_20250913")
    plays_path = out / "plays.jsonl"
    audit_dir  = out / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    if not plays_path.exists():
        sys.exit(f"[error] {plays_path} not found")

    plays = [json.loads(x) for x in plays_path.read_text().splitlines() if x.strip()]

    kept = {"offense": [], "defense": []}
    debug_rows = []
    for i, p in enumerate(plays):
        idx = p.get("index") or p.get("idx") or (i+1)
        side = (p.get("side") or "").lower()
        reasons = []
        keep = keep_for_analytics(p, reasons)
        rp = rp_used(p)
        diru = dir_used(p, rp)
        row = {
            "index": idx,
            "kept": str(bool(keep)).lower(),
            "exclude_reasons": "|".join(reasons),
            "side": side or "unknown",
            "rp_used": rp,
            "rp_flags": rp_from_flags(p),
            "dir_used": diru,
            "run_dir": (p.get("run_dir") or "unknown"),
            "direction": (p.get("direction") or "unknown"),
            "phase": (p.get("phase") or "unknown"),
            "st_fix": (p.get("st_fix") or p.get("st") or ""),
            "exclude_flag": str(bool(p.get("exclude_from_analytics"))).lower(),
            "src": (p.get("src") or ""),
            "title": (p.get("title") or ""),
        }
        debug_rows.append(row)
        if keep and side in kept:
            kept[side].append({"index": idx, "rp": rp, "dir": diru})

    # print kept counts
    print("[kept counts]")
    for s in ("offense","defense"):
        print(f"  {s}: {len(kept[s])}")

    # write debug
    dbg_p = audit_dir / "audit_kept_debug.csv"
    with dbg_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()) if debug_rows else [
            "index","kept","exclude_reasons","side","rp_used","rp_flags","dir_used",
            "run_dir","direction","phase","st_fix","exclude_flag","src","title"
        ])
        w.writeheader(); w.writerows(debug_rows)

    # build summary (side,bucket,value,count) to mirror quick_* CSVs
    summary_rows = []
    for side, items in kept.items():
        cnt = collections.Counter()
        for it in items:
            rp = it["rp"]
            cnt[("rp", rp)] += 1
            cnt[("rp_dir", f"{rp}:{it['dir']}")] += 1
        for (bucket, value), c in sorted(cnt.items()):
            summary_rows.append({"side": side, "bucket": bucket, "value": value, "count": c})

    sum_p = audit_dir / "audit_summary.csv"
    with sum_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["side","bucket","value","count"])
        w.writeheader(); w.writerows(summary_rows)

    # disagreements = where rp_used != rp_flags OR (rp_used=='run' and dir_used != run_dir)
    # OR (rp_used=='pass' and dir_used != direction)
    dis_rows = []
    for r in debug_rows:
        rp = r["rp_used"]; flags = r["rp_flags"]
        diru = r["dir_used"]; run_dir = (r["run_dir"] or "unknown").lower()
        direc = (r["direction"] or "unknown").lower()
        dir_mismatch = (rp=="run" and diru!=norm(run_dir, {"left","right","unknown"})) or \
                       (rp=="pass" and diru!=norm(direc, {"left","right","unknown"}))
        if rp != flags or dir_mismatch:
            dis_rows.append(r)
    dis_p = audit_dir / "audit_disagreements.csv"
    with dis_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()) if debug_rows else [])
        if debug_rows: w.writeheader()
        w.writerows(dis_rows)

    print(f"[wrote] {dbg_p}")
    print(f"[wrote] {sum_p}")
    print(f"[wrote] {dis_p}")

if __name__ == "__main__":
    main()
