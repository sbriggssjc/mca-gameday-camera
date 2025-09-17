from pathlib import Path
import sys, json, csv, collections, re

ST_EXCLUDE = {"xp","kickoff","kick","punt","return","kneel","spike"}

# ---------- filters (match analytics) ----------
def keep_for_analytics(p):
    if p.get("exclude_from_analytics"):
        return False, ["exclude_flag"]

    st = (p.get("st_fix") or p.get("st") or "").strip().lower()
    if st in ST_EXCLUDE:
        return False, [f"st:{st}"]
    if p.get("special_teams"):
        return False, ["special_teams"]

    # Allow "", "unknown"; only drop explicit non-live phases
    ph = (p.get("phase") or "").strip().lower()
    if ph in {"dead","deadball","pre","presnap","post","postplay","timeout","halftime","setup"}:
        return False, [f"phase:{ph}"]

    if p.get("side") not in {"offense","defense"}:
        return False, [f"side:{p.get('side')}"]

    return True, []

# ---------- RP & Direction helpers ----------
def rp_from_flags(p):
    # Prefer pass if both are accidentally true
    if p.get("is_pass"): return "pass"
    if p.get("is_run"):  return "run"
    return "unknown"

def parse_rp_dir_field(s):
    s = (s or "").strip().lower()
    m = re.match(r'^(run|pass)\s*:\s*(left|right|unknown)\b', s)
    if m:
        return m.group(1), m.group(2)
    return None, None

def rp_final(p):
    # 1) Trust the resolved field from audit/sync (this is what quick_* uses)
    for k in ("rp_used","rp_fix","rp"):
        v = (p.get(k) or "").strip().lower()
        if v in {"run","pass"}:
            return v
    # 2) Try rp_dir string
    rp2, _ = parse_rp_dir_field(p.get("rp_dir"))
    if rp2 in {"run","pass"}:
        return rp2
    # 3) Fall back to flags
    return rp_from_flags(p)

def norm_dir(v: str):
    s = (v or "").strip().lower()
    if s in {"l","left"} or "left" in s:   return "left"
    if s in {"r","right"} or "right" in s: return "right"
    return "unknown"

def dir_final(p, rp):
    # 1) Trust the resolved field from audit/sync if present
    d_used = (p.get("dir_used") or "").strip().lower()
    if d_used in {"left","right","unknown"}:
        return d_used

    # 2) If rp_dir encodes both and RP matches, use it
    rp2, d2 = parse_rp_dir_field(p.get("rp_dir"))
    if rp2 == rp and d2 in {"left","right","unknown"}:
        return d2

    # 3) Otherwise best-available per RP
    if rp == "run":
        for k in ("run_dir","dir_fix","dir","direction","flow_dir","off_dir"):
            d = p.get(k)
            if d: return norm_dir(d)
        return "unknown"
    elif rp == "pass":
        for k in ("direction","dir_fix","dir","flow_dir","pass_dir"):
            d = p.get(k)
            if d: return norm_dir(d)
        return "unknown"
    else:
        for k in ("direction","dir_fix","dir","run_dir","flow_dir"):
            d = p.get(k)
            if d: return norm_dir(d)
        return "unknown"

# ---------- main ----------
def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/opponent_jenks_silver_20250913")
    plays_path = out / "plays.jsonl"
    audit_dir  = out / "audit"
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

        rp_f  = rp_final(p)
        rp_fl = rp_from_flags(p)
        d_f   = dir_final(p, rp_f)

        if keep:
            kept.append((idx, p))
            by_side_counts[p["side"]] += 1

        debug_rows.append({
            "index": idx,
            "kept": str(bool(keep)).lower(),
            "exclude_reasons": ";".join(reasons) if reasons else "",
            "side": p.get("side",""),
            "rp_used": (p.get("rp_used") or ""),
            "rp_flags": rp_fl,
            "rp_final": rp_f,
            "dir_used": (p.get("dir_used") or ""),
            "run_dir": p.get("run_dir",""),
            "direction": p.get("direction",""),
            "dir": p.get("dir",""),
            "dir_fix": p.get("dir_fix",""),
            "phase": p.get("phase",""),
            "st_fix": p.get("st_fix",""),
            "exclude_from_analytics": str(bool(p.get("exclude_from_analytics"))).lower(),
            "src": p.get("src",""),
            "title": p.get("title",""),
        })

        # disagreements: either filtered-out (show why) or RP conflict on kept
        if keep and (rp_f != "unknown") and (rp_fl != "unknown") and rp_f != rp_fl:
            disagree_rows.append([
                idx, "true", "",
                p.get("side",""),
                rp_f, rp_fl,
                d_f, p.get("run_dir",""), p.get("direction",""),
                p.get("phase",""), p.get("st_fix",""),
                str(bool(p.get("exclude_from_analytics"))).lower(),
                p.get("src",""), p.get("title","")
            ])
        elif not keep:
            disagree_rows.append([
                idx, "false", ";".join(reasons),
                p.get("side",""),
                rp_f, rp_fl,
                d_f, p.get("run_dir",""), p.get("direction",""),
                p.get("phase",""), p.get("st_fix",""),
                str(bool(p.get("exclude_from_analytics"))).lower(),
                p.get("src",""), p.get("title","")
            ])

    # Summary (mirror quick_*) – counts by side and (rp / rp_dir)
    cnt = collections.Counter()
    for _, p in kept:
        side = p["side"]
        rp   = rp_final(p)
        d    = dir_final(p, rp)
        cnt[(side,"rp",rp)] += 1
        cnt[(side,"rp_dir",f"{rp}:{d}")] += 1

    with (audit_dir/"audit_kept_debug.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
        w.writeheader(); w.writerows(debug_rows)

    with (audit_dir/"audit_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side","bucket","value","count"])
        for (side,bucket,value),c in sorted(cnt.items()):
            w.writerow([side,bucket,value,c])

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
