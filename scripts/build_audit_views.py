from pathlib import Path
import sys, json, csv, collections

# --- helpers ---------------------------------------------------------------

ST_EXCLUDE = {"xp","kickoff","kick","punt","return","kneel","spike"}

def keep_for_analytics(p):
    # explicit excludes
    if p.get("exclude_from_analytics"):
        return False, ["exclude_flag"]

    # special teams by tag or st/st_fix
    st = (p.get("st_fix") or p.get("st") or "").strip().lower()
    if st in ST_EXCLUDE:
        return False, [f"st:{st}"]
    if p.get("special_teams"):
        return False, ["special_teams"]

    # phase: allow "", "unknown"; only drop explicit non-live phases
    ph = (p.get("phase") or "").strip().lower()
    if ph in {"dead","deadball","pre","presnap","post","postplay","timeout","halftime","setup"}:
        return False, [f"phase:{ph}"]

    # only analyze offense/defense
    if p.get("side") not in {"offense","defense"}:
        return False, [f"side:{p.get('side')}"]

    return True, []

def rp_from_flags(p):
    # derive from is_run/is_pass flags (used only as fallback)
    if p.get("is_run"):
        return "run"
    if p.get("is_pass"):
        return "pass"
    return "unknown"

def rp_final(p):
    # prefer the FINAL label used by analytics/quick CSVs
    rp = (p.get("rp") or "").strip().lower()
    if rp in {"run","pass"}:
        return rp
    return rp_from_flags(p)

def dir_final(p, rp):
    if rp == "run":
        d = (p.get("run_dir") or p.get("direction") or "unknown").strip().lower()
    elif rp == "pass":
        d = (p.get("direction") or "unknown").strip().lower()
    else:
        d = "unknown"
    # normalize to left/right/unknown for stable buckets
    if d not in {"left","right"}:
        d = "unknown"
    return d

# --- main ------------------------------------------------------------------

def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/opponent_jenks_silver_20250913")
    plays_path = out / "plays.jsonl"
    audit_dir = out / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    if not plays_path.exists():
        raise SystemExit(f"[error] plays.jsonl not found at {plays_path}")

    plays = [json.loads(x) for x in plays_path.read_text().splitlines() if x.strip()]

    kept = []
    debug_rows = []
    disagree_rows = []
    by_side_counts = {"offense":0, "defense":0}

    for idx, p in enumerate(plays, start=1):
        keep, reasons = keep_for_analytics(p)
        if keep:
            kept.append((idx, p))
            by_side_counts[p["side"]] += 1
        debug_rows.append({
            "index": idx,
            "kept": str(bool(keep)).lower(),
            "exclude_reasons": ";".join(reasons) if reasons else "",
            "side": p.get("side",""),
            "rp": p.get("rp",""),
            "is_run": str(bool(p.get("is_run"))).lower(),
            "is_pass": str(bool(p.get("is_pass"))).lower(),
            "run_dir": p.get("run_dir",""),
            "direction": p.get("direction",""),
            "phase": p.get("phase",""),
            "st_fix": p.get("st_fix",""),
            "src": p.get("src",""),
            "title": p.get("title",""),
        })

        # disagreement = final rp vs flags (only for kept clips)
        rp_f = rp_final(p)
        rp_fl = rp_from_flags(p)
        if keep and (rp_f != "unknown") and (rp_fl != "unknown") and rp_f != rp_fl:
            disagree_rows.append([
                idx, "true", "",
                p.get("side",""),
                rp_f, rp_fl,
                dir_final(p, rp_f), p.get("run_dir",""), p.get("direction",""),
                p.get("phase",""), p.get("st_fix",""), str(bool(p.get("exclude_from_analytics"))).lower(),
                p.get("src",""), p.get("title","")
            ])
        elif not keep:
            # show why it was dropped
            disagree_rows.append([
                idx, "false", ";".join(reasons),
                p.get("side",""),
                rp_final(p), rp_from_flags(p),
                dir_final(p, rp_final(p)), p.get("run_dir",""), p.get("direction",""),
                p.get("phase",""), p.get("st_fix",""), str(bool(p.get("exclude_from_analytics"))).lower(),
                p.get("src",""), p.get("title","")
            ])

    # Summary (mirror quick_* CSVs)
    cnt = collections.Counter()
    for _, p in kept:
        side = p["side"]
        rp = rp_final(p)
        cnt[(side, "rp", rp)] += 1
        cnt[(side, "rp_dir", f"{rp}:{dir_final(p, rp)}")] += 1

    # Write files
    # kept debug
    with (audit_dir/"audit_kept_debug.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
        w.writeheader(); w.writerows(debug_rows)

    # summary
    with (audit_dir/"audit_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side","bucket","value","count"])
        for (side,bucket,value), c in sorted(cnt.items()):
            w.writerow([side,bucket,value,c])

    # disagreements
    with (audit_dir/"audit_disagreements.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index","kept","exclude_reasons","side","rp_used","rp_flags","dir_used",
                    "run_dir","direction","phase","st_fix","exclude_flag","src","title"])
        w.writerows(disagree_rows)

    print("[kept counts]")
    for s in ("offense","defense"):
        print(f"  {s}: {by_side_counts[s]}")
    print(f"[wrote] {audit_dir/'audit_kept_debug.csv'}")
    print(f"[wrote] {audit_dir/'audit_summary.csv'}")
    print(f"[wrote] {audit_dir/'audit_disagreements.csv'}")

if __name__ == "__main__":
    main()
