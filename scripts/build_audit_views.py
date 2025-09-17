from pathlib import Path
import sys, json, csv, collections

ST_EXCLUDE   = {"xp","kickoff","punt","return","kneel","spike","kick"}
NON_LIVE     = {"dead","deadball","pre","presnap","post","postplay","timeout","halftime","setup"}

def keep_for_analytics(p):
    # Explicit exclude
    if p.get("exclude_from_analytics"):
        return False, ["exclude_flag"]
    # Special teams (via st/st_fix) or explicit flag
    st = (p.get("st_fix") or p.get("st") or "").strip().lower()
    if st in ST_EXCLUDE:
        return False, [f"st:{st}"]
    if p.get("special_teams"):
        return False, ["special_teams"]
    # Phase: only exclude clearly non-live phases; keep "", "unknown", etc.
    ph = (p.get("phase") or "").strip().lower()
    if ph in NON_LIVE:
        return False, [f"phase:{ph}"]
    # Side must be offense/defense
    side = p.get("side")
    if side not in {"offense","defense"}:
        return False, [f"side:{side}"]
    return True, []

def rp_used_of(p):
    if p.get("is_run"):  return "run"
    if p.get("is_pass"): return "pass"
    return "unknown"

def rp_flags_of(p):
    # If an original auto tag exists (e.g., "rp"), prefer it; otherwise fall back to current flags
    rp = (p.get("rp") or "").strip().lower()
    if rp in {"run","pass"}: return rp
    return rp_used_of(p)

def dir_used_of(p):
    ru = rp_used_of(p)
    if ru == "run":
        return (p.get("run_dir") or "unknown").lower()
    if ru == "pass":
        return (p.get("direction") or "unknown").lower()
    return "unknown"

def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/opponent_jenks_silver_20250913")
    plays_path = out / "plays.jsonl"
    audit_dir  = out / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    if not plays_path.exists():
        print(f"[error] {plays_path} not found")
        sys.exit(1)

    plays = [json.loads(l) for l in plays_path.read_text().splitlines() if l.strip()]

    kept = []
    debug_rows = []
    for i, p in enumerate(plays):
        k, reasons = keep_for_analytics(p)
        if k: kept.append((i, p))
        debug_rows.append({
            "index": i,
            "kept": str(k).lower(),
            "exclude_reasons": ";".join(reasons),
            "side": p.get("side",""),
            "rp_used": rp_used_of(p),
            "rp_flags": rp_flags_of(p),
            "dir_used": dir_used_of(p),
            "run_dir": (p.get("run_dir") or ""),
            "direction": (p.get("direction") or ""),
            "phase": (p.get("phase") or ""),
            "st_fix": (p.get("st_fix") or p.get("st") or ""),
            "exclude_flag": str(bool(p.get("exclude_from_analytics"))).lower(),
            "src": p.get("src",""),
            "title": p.get("title",""),
        })

    # Write debug
    dbg_p = audit_dir / "audit_kept_debug.csv"
    with dbg_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
        w.writeheader(); w.writerows(debug_rows)

    # Summary (mirror quick_* CSVs): counts by side x (rp, rp_dir)
    cnt = collections.Counter()
    for _, p in kept:
        side = p["side"]
        ru = rp_used_of(p)
        cnt[(side,"rp",ru)] += 1
        d = dir_used_of(p)
        cnt[(side,"rp_dir",f"{ru}:{d}")] += 1

    sum_p = audit_dir / "audit_summary.csv"
    with sum_p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side","bucket","value","count"])
        for (side,bucket,value), c in sorted(cnt.items()):
            w.writerow([side,bucket,value,c])

    # Disagreements between rp_used and rp_flags (good retraining examples)
    dis_p = audit_dir / "audit_disagreements.csv"
    with dis_p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index","kept","exclude_reasons","side","rp_used","rp_flags","dir_used",
                    "run_dir","direction","phase","st_fix","exclude_flag","src","title"])
        for i, p in kept:
            ru = rp_used_of(p)
            rf = rp_flags_of(p)
            if ru != rf:
                # find its debug row to grab reasons/flags consistently
                dr = next((r for r in debug_rows if int(r["index"])==i), None)
                w.writerow([
                    i,
                    dr["kept"] if dr else "true",
                    dr["exclude_reasons"] if dr else "",
                    p.get("side",""),
                    ru,
                    rf,
                    dir_used_of(p),
                    p.get("run_dir",""),
                    p.get("direction",""),
                    p.get("phase",""),
                    (p.get("st_fix") or p.get("st") or ""),
                    str(bool(p.get("exclude_from_analytics"))).lower(),
                    p.get("src",""),
                    p.get("title",""),
                ])

    print("[kept counts]")
    by_side = collections.Counter(p.get("side") for _, p in kept)
    print(f"  offense: {by_side.get('offense',0)}")
    print(f"  defense: {by_side.get('defense',0)}")
    print(f"[wrote] {dbg_p}")
    print(f"[wrote] {sum_p}")
    print(f"[wrote] {dis_p}")

if __name__ == "__main__":
    main()
