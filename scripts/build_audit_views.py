cat > scripts/build_audit_views.py <<'PY'
from pathlib import Path
import sys, json, csv, collections, re

ST_EXCLUDE = {"xp","kickoff","kick","punt","return","kneel","spike"}

def keep_for_analytics(p):
    if p.get("exclude_from_analytics"): return False, ["exclude_flag"]
    st = (p.get("st_fix") or p.get("st") or "").strip().lower()
    if st in ST_EXCLUDE: return False, [f"st:{st}"]
    if p.get("special_teams"): return False, ["special_teams"]
    ph = (p.get("phase") or "").strip().lower()
    # keep unknown/empty; only exclude explicit non-live phases
    if ph in {"dead","deadball","pre","presnap","post","postplay","timeout","halftime","setup"}:
        return False, [f"phase:{ph}"]
    if p.get("side") not in {"offense","defense"}: return False, [f"side:{p.get('side')}"]
    return True, []

def rp_from_flags(p):
    if p.get("is_pass"): return "pass"
    if p.get("is_run"):  return "run"
    return "unknown"

def parse_rp_dir_field(s):
    s = (s or "").strip().lower()
    m = re.match(r'^(run|pass)\s*:\s*(left|right|unknown)\b', s)
    if m: return m.group(1), m.group(2)
    return None, None

def rp_final(p):
    for k in ("rp_used","rp_fix","rp"):
        v = (p.get(k) or "").strip().lower()
        if v in {"run","pass"}: return v
    rp2, _ = parse_rp_dir_field(p.get("rp_dir"))
    if rp2 in {"run","pass"}: return rp2
    return rp_from_flags(p)

def norm_dir(v):
    s = (v or "").strip().lower()
    if "left" in s or s in {"l","left"}: return "left"
    if "right" in s or s in {"r","right"}: return "right"
    return "unknown"

def dir_final(p, rp):
    d_used = (p.get("dir_used") or "").strip().lower()
    if d_used in {"left","right","unknown"}: return d_used
    rp2, d2 = parse_rp_dir_field(p.get("rp_dir"))
    if rp2 == rp and d2 in {"left","right","unknown"}: return d2
    if rp == "run":
        for k in ("run_dir","dir_fix","dir","direction","flow_dir","off_dir"):
            d = p.get(k)
            if d: return norm_dir(d)
        return "unknown"
    if rp == "pass":
        for k in ("direction","dir_fix","dir","flow_dir","pass_dir"):
            d = p.get(k)
            if d: return norm_dir(d)
        return "unknown"
    for k in ("direction","dir_fix","dir","run_dir","flow_dir"):
        d = p.get(k)
        if d: return norm_dir(d)
    return "unknown"

def load_quick_counts(out_dir: Path):
    quick = {}
    for fname in ("quick_tendencies_offense.csv","quick_tendencies_defense.csv"):
        p = out_dir/fname
        if not p.exists():
            return None
        for r in csv.DictReader(p.open()):
            quick[(r["side"], r["bucket"], r["value"])] = int(r["count"])
    return quick

def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/opponent_jenks_silver_20250913")
    plays_path = out/"plays.jsonl"
    audit_dir = out/"audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    if not plays_path.exists():
        raise SystemExit(f"[error] plays.jsonl not found at {plays_path}")

    plays = [json.loads(x) for x in plays_path.read_text().splitlines() if x.strip()]

    kept = []
    debug_rows, disagree_rows = [], []
    kept_counts = {"offense":0, "defense":0}

    for idx, p in enumerate(plays, start=1):
        keep, reasons = keep_for_analytics(p)
        rp_f  = rp_final(p)
        rp_fl = rp_from_flags(p)
        d_f   = dir_final(p, rp_f)

        if keep:
            kept.append((idx, p))
            kept_counts[p.get("side","")] += 1

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

        # disagreements (model flags vs final)
        if keep and (rp_f != "unknown") and (rp_fl != "unknown") and rp_f != rp_fl:
            disagree_rows.append([idx,"true","",p.get("side",""),rp_f,rp_fl,d_f,
                                  p.get("run_dir",""),p.get("direction",""),
                                  p.get("phase",""),p.get("st_fix",""),
                                  str(bool(p.get("exclude_from_analytics"))).lower(),
                                  p.get("src",""),p.get("title","")])
        elif not keep:
            disagree_rows.append([idx,"false",";".join(reasons),p.get("side",""),rp_f,rp_fl,d_f,
                                  p.get("run_dir",""),p.get("direction",""),
                                  p.get("phase",""),p.get("st_fix",""),
                                  str(bool(p.get("exclude_from_analytics"))).lower(),
                                  p.get("src",""),p.get("title","")])

    # Summary: prefer quick_* if present (authoritative for analytics)
    quick_counts = load_quick_counts(out)
    rows_summary = []
    if quick_counts:
        for (side,bucket,value), c in sorted(quick_counts.items()):
            rows_summary.append([side,bucket,value,c])
    else:
        # Fallback: compute from kept
        cnt = collections.Counter()
        for _, p in kept:
            side = p["side"]
            rp   = rp_final(p)
            d    = dir_final(p, rp)
            cnt[(side,"rp",rp)] += 1
            cnt[(side,"rp_dir",f"{rp}:{d}")] += 1
        for (side,bucket,value), c in sorted(cnt.items()):
            rows_summary.append([side,bucket,value,c])

    # write files
    with (audit_dir/"audit_kept_debug.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
        w.writeheader(); w.writerows(debug_rows)

    with (audit_dir/"audit_summary.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["side","bucket","value","count"]); w.writerows(rows_summary)

    with (audit_dir/"audit_disagreements.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index","kept","exclude_reasons","side","rp_used","rp_flags","dir_used",
                    "run_dir","direction","phase","st_fix","exclude_flag","src","title"])
        w.writerows(disagree_rows)

    print("[kept counts]")
    print(f"  offense: {kept_counts['offense']}")
    print(f"  defense: {kept_counts['defense']}")
    print(f"[wrote] {audit_dir/'audit_kept_debug.csv'}")
    print(f"[wrote] {audit_dir/'audit_summary.csv'}")
    print(f"[wrote] {audit_dir/'audit_disagreements.csv'}")

if __name__ == "__main__":
    main()
PY