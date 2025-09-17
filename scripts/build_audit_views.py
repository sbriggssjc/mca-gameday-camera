from pathlib import Path
import sys, json, csv, collections

ST_EXCLUDE = {"xp","kickoff","kick","punt","return","kneel","spike"}

def keep_for_analytics(p):
    if p.get("exclude_from_analytics"): return False, ["exclude_flag"]
    st = (p.get("st_fix") or p.get("st") or "").lower()
    if st in ST_EXCLUDE: return False, [f"st:{st}"]
    if p.get("special_teams"): return False, ["special_teams"]
    ph = (p.get("phase") or "").lower()
    if ph and ph not in {"live","play"}: return False, [f"phase:{ph}"]
    if p.get("side") not in {"offense","defense"}: return False, [f"side:{p.get('side')}"]
    return True, []

def rp_and_dir(p):
    # 1) unified rp_dir if available (e.g., "pass:left", "run:right")
    rpd = p.get("rp_dir") or p.get("rpdir")
    if isinstance(rpd, str) and ":" in rpd:
        rp, d = rpd.split(":", 1)
        rp, d = (rp or "unknown").strip(), (d or "unknown").strip()
        if rp in {"run","pass"}:
            return rp, d

    # 2) fall back to rp_fix / rp + run_dir/direction
    rp_fix = p.get("rp_fix")
    if rp_fix in {"run","pass"}:
        d = (p.get("run_dir") if rp_fix == "run" else p.get("direction")) or "unknown"
        return rp_fix, d

    rp = p.get("rp")
    if rp in {"run","pass"}:
        d = (p.get("run_dir") if rp == "run" else p.get("direction")) or "unknown"
        return rp, d

    # 3) booleans
    if p.get("is_run"):
        return "run", (p.get("run_dir") or "unknown")
    if p.get("is_pass"):
        return "pass", (p.get("direction") or "unknown")

    return "unknown", "unknown"

def rp_flags_raw(p):
    if p.get("is_run"): return "run"
    if p.get("is_pass"): return "pass"
    return "unknown"

def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/opponent_jenks_silver_20250913")
    plays_path = out / "plays.jsonl"
    audit_dir = out / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    if not plays_path.exists():
        sys.exit(f"[error] {plays_path} not found")

    plays = [json.loads(x) for x in plays_path.read_text().splitlines() if x.strip()]

    kept_by_side = {"offense": [], "defense": []}
    kept_debug_rows = []
    for idx, p in enumerate(plays, start=1):
        keep, reasons = keep_for_analytics(p)
        rp_used, dir_used = rp_and_dir(p)
        rp_flags = rp_flags_raw(p)

        kept_debug_rows.append({
            "index": idx,
            "kept": str(bool(keep)).lower(),
            "exclude_reasons": ";".join(reasons),
            "side": p.get("side") or "unknown",
            "rp_used": rp_used,
            "rp_flags": rp_flags,
            "dir_used": dir_used,
            "run_dir": p.get("run_dir") or "",
            "direction": p.get("direction") or "",
            "phase": p.get("phase") or "",
            "st_fix": (p.get("st_fix") or p.get("st") or ""),
            "exclude_flag": str(bool(p.get("exclude_from_analytics"))).lower(),
            "src": p.get("src") or "",
            "title": p.get("title") or "",
        })

        if keep:
            s = p.get("side")
            if s in kept_by_side:
                kept_by_side[s].append({"rp": rp_used, "dir": dir_used})

    # Write debug rows
    debug_path = audit_dir / "audit_kept_debug.csv"
    with debug_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kept_debug_rows[0].keys()) if kept_debug_rows else [])
        if kept_debug_rows: w.writeheader(); w.writerows(kept_debug_rows)

    # Summary counts (mirror quick_* layout)
    summary_rows = []
    for side, arr in kept_by_side.items():
        cnt = collections.Counter()
        for x in arr:
            rp = x["rp"]
            d  = x["dir"] or "unknown"
            cnt[("rp", rp)] += 1
            cnt[("rp_dir", f"{rp}:{d}")] += 1
        for (bucket, value), c in sorted(cnt.items()):
            summary_rows.append({"side": side, "bucket": bucket, "value": value, "count": c})

    summary_path = audit_dir / "audit_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["side","bucket","value","count"])
        w.writeheader(); w.writerows(summary_rows)

    # Disagreements: rp_used vs rp_flags
    dis_rows = []
    for r in kept_debug_rows:
        if r["kept"] == "true" and r["rp_used"] != r["rp_flags"]:
            dis_rows.append({k: r[k] for k in [
                "index","side","rp_used","rp_flags","dir_used","run_dir","direction",
                "phase","st_fix","exclude_flag","src","title"
            ]})
    disagree_path = audit_dir / "audit_disagreements.csv"
    with disagree_path.open("w", newline="") as f:
        hdr = ["index","side","rp_used","rp_flags","dir_used","run_dir","direction","phase","st_fix","exclude_flag","src","title"]
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader(); w.writerows(dis_rows)

    print("[kept counts]")
    for side in ("offense","defense"):
        print(f"  {side}: {len(kept_by_side[side])}")
    print(f"[wrote] {debug_path}")
    print(f"[wrote] {summary_path}")
    print(f"[wrote] {disagree_path}")

if __name__ == "__main__":
    main()
