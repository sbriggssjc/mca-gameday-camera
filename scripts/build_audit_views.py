from pathlib import Path
import sys, json, csv, collections

def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/opponent_jenks_silver_20250913")
    plays_path = out / "plays.jsonl"
    audit = out / "audit"
    audit.mkdir(parents=True, exist_ok=True)

    if not plays_path.exists():
        raise SystemExit(f"[error] {plays_path} not found")

    plays = [json.loads(l) for l in plays_path.read_text().splitlines() if l.strip()]

    # Same exclusions as analytics
    ST_EXCLUDE = {"xp","kickoff","punt","return","kneel","spike"}

    def keep(p):
        if p.get("exclude_from_analytics"):
            return False, ["exclude_flag"]
        st = (p.get("st_fix") or p.get("st") or "").lower()
        if st in ST_EXCLUDE:
            return False, [f"st:{st}"]
        if p.get("special_teams"):
            return False, ["special_teams"]
        ph = (p.get("phase") or "").lower()
        if ph and ph not in {"live","play"}:
            return False, [f"phase:{ph}"]
        if p.get("side") not in {"offense","defense"}:
            return False, [f"side:{p.get('side')}"]
        return True, []

    def rp_dir(p):
        # precedence: *_fix > high-level auto > flags
        rp = (p.get("rp_fix") or p.get("rp") or
              ("run" if p.get("is_run") else "pass" if p.get("is_pass") else "unknown")).lower()

        if rp == "run":
            d = (p.get("run_dir_fix") or p.get("run_dir") or "unknown").lower()
        elif rp == "pass":
            d = (p.get("dir_fix") or p.get("direction") or "unknown").lower()
        else:
            d = (p.get("dir_fix") or p.get("direction") or p.get("run_dir") or "unknown").lower()

        # normalize direction a bit
        if d not in {"left","right","middle","unknown"}:
            d = ("left" if "left" in d else
                 "right" if "right" in d else
                 "middle" if "mid" in d else
                 "unknown")
        return rp, d

    debug_rows = []
    by_side = {"offense": [], "defense": []}

    for p in plays:
        kept, reasons = keep(p)
        rp_used, dir_used = rp_dir(p)
        rp_flags = "run" if p.get("is_run") else ("pass" if p.get("is_pass") else "unknown")
        debug_rows.append([
            p.get("index"), p.get("side"), p.get("phase"),
            p.get("st"), p.get("st_fix"),
            bool(p.get("exclude_from_analytics")), bool(p.get("special_teams")),
            bool(p.get("is_run")), bool(p.get("is_pass")),
            p.get("rp_fix"), p.get("rp"), rp_flags,
            p.get("dir_fix"), p.get("run_dir"), p.get("direction"),
            rp_used, dir_used, kept, "|".join(reasons),
            p.get("src") or p.get("clip") or "", p.get("title") or ""
        ])
        if kept and p.get("side") in by_side:
            by_side[p["side"]].append((rp_used, dir_used))

    # Debug CSV: why each play was/wasn't kept + which labels were used
    with (audit/"audit_kept_debug.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "index","side","phase","st","st_fix","exclude_flag","special_teams",
            "is_run","is_pass","rp_fix","rp_auto","rp_flags",
            "dir_fix","run_dir","direction",
            "rp_used","dir_used","kept","exclude_reasons","src","title"
        ])
        w.writerows(debug_rows)

    # Summary CSV: counts by side/rp and side/rp_dir (should match quick_* CSVs)
    with (audit/"audit_summary.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["side","bucket","value","count"])
        for side, vals in by_side.items():
            cnt = collections.Counter()
            for rp, d in vals:
                cnt[("rp", rp)] += 1
                cnt[("rp_dir", f"{rp}:{d}")] += 1
            for (bucket, value), n in sorted(cnt.items()):
                w.writerow([side, bucket, value, n])

    # Disagreements CSV: useful training pairs (final rp_used vs flags)
    with (audit/"audit_disagreements.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index","side","rp_used","rp_flags","dir_used","run_dir","direction","phase","st_fix","exclude_flag","src","title"])
        for row, p in zip(debug_rows, plays):
            kept = row[17]
            if not kept:
                continue
            rp_used, dir_used = row[15], row[16]
            rp_flags = row[11]
            if rp_used != rp_flags:
                w.writerow([
                    row[0], row[1], rp_used, rp_flags, dir_used,
                    p.get("run_dir"), p.get("direction"), p.get("phase"), p.get("st_fix"),
                    row[5], p.get("src") or "", p.get("title") or ""
                ])

    # Console sanity
    print("[kept counts]")
    for side, vals in by_side.items():
        print(f"  {side}: {len(vals)}")
    print("[wrote]", audit/"audit_kept_debug.csv")
    print("[wrote]", audit/"audit_summary.csv")
    print("[wrote]", audit/"audit_disagreements.csv")

if __name__ == "__main__":
    main()
