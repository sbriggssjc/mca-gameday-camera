from pathlib import Path
import sys, os, json, csv, collections

ST_EXCLUDE = {"xp","kickoff","kick","punt","return","kneel","spike"}

def keep_for_analytics(p):
    # Match your analytics rules, but be permissive on 'phase'
    if p.get("exclude_from_analytics"):
        return False, ["exclude_flag"]
    st = (p.get("st_fix") or p.get("st") or "").lower().strip()
    if st in ST_EXCLUDE:
        return False, [f"st:{st}"]
    if p.get("special_teams"):
        return False, ["special_teams"]
    # DON'T hard filter on phase; many valid plays have blank/misc phases
    side = p.get("side")
    if side not in {"offense","defense"}:
        return False, [f"side:{side}"]
    return True, []

def rp_and_dir(p):
    # Final label with fallbacks
    rp = (p.get("rp_fix") or p.get("rp") or
          ("run" if p.get("is_run") else ("pass" if p.get("is_pass") else "unknown"))).lower()
    if rp == "run":
        d = (p.get("run_dir_fix") or p.get("run_dir") or "unknown").lower()
    elif rp == "pass":
        d = (p.get("dir_fix") or p.get("direction") or "unknown").lower()
    else:
        d = (p.get("dir_fix") or p.get("direction") or p.get("run_dir") or "unknown").lower()
    # normalize
    if d not in {"left","right","middle","unknown"}:
        d = ("left" if "left" in d else
             "right" if "right" in d else
             "middle" if "mid" in d else
             "unknown")
    return rp, d

def main():
    arg = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    out = Path(arg or os.environ.get("OUT", "output/opponent_jenks_silver_20250913"))
    plays_path = out / "plays.jsonl"
    audit_dir = out / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    if not plays_path.exists():
        raise SystemExit(f"[error] {plays_path} not found")

    plays = [json.loads(l) for l in plays_path.read_text().splitlines() if l.strip()]

    kept_rows = {"offense": [], "defense": []}
    debug_rows = []

    for p in plays:
        kept, reasons = keep_for_analytics(p)
        rp_used, dir_used = rp_and_dir(p)
        rp_flags = "run" if p.get("is_run") else ("pass" if p.get("is_pass") else "unknown")
        debug_rows.append([
            p.get("index"), p.get("side"),
            p.get("phase"), p.get("st"), p.get("st_fix"),
            bool(p.get("exclude_from_analytics")), bool(p.get("special_teams")),
            bool(p.get("is_run")), bool(p.get("is_pass")),
            p.get("rp_fix"), p.get("rp"), rp_flags,
            p.get("dir_fix"), p.get("run_dir"), p.get("direction"),
            rp_used, dir_used, kept, "|".join(reasons),
            p.get("src") or p.get("clip") or "", p.get("title") or ""
        ])
        if kept and p.get("side") in kept_rows:
            kept_rows[p["side"]].append((rp_used, dir_used))

    # Debug
    with (audit_dir/"audit_kept_debug.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "index","side","phase","st","st_fix","exclude_flag","special_teams",
            "is_run","is_pass","rp_fix","rp_auto","rp_flags",
            "dir_fix","run_dir","direction",
            "rp_used","dir_used","kept","exclude_reasons","src","title"
        ])
        w.writerows(debug_rows)

    # Summary
    with (audit_dir/"audit_summary.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["side","bucket","value","count"])
        for side, vals in kept_rows.items():
            cnt = collections.Counter()
            for rp, d in vals:
                cnt[("rp", rp)] += 1
                cnt[("rp_dir", f"{rp}:{d}")] += 1
            for (bucket, value), n in sorted(cnt.items()):
                w.writerow([side, bucket, value, n])

    # Disagreements (rp_used vs raw flags)
    with (audit_dir/"audit_disagreements.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index","side","rp_used","rp_flags","dir_used","run_dir","direction","phase","st_fix","exclude_flag","src","title"])
        for row, p in zip(debug_rows, plays):
            if not row[17]:  # kept == False? skip; we care about included analytics plays
                continue
            rp_used, dir_used, rp_flags = row[15], row[16], row[11]
            if rp_used != rp_flags:
                w.writerow([
                    row[0], row[1], rp_used, rp_flags, dir_used,
                    p.get("run_dir"), p.get("direction"), p.get("phase"), p.get("st_fix"),
                    row[5], p.get("src") or "", p.get("title") or ""
                ])

    print("[kept counts]")
    for side, vals in kept_rows.items():
        print(f"  {side}: {len(vals)}")
    print("[wrote]", audit_dir/"audit_kept_debug.csv")
    print("[wrote]", audit_dir/"audit_summary.csv")
    print("[wrote]", audit_dir/"audit_disagreements.csv")

if __name__ == "__main__":
    main()
