#!/usr/bin/env python3
from pathlib import Path
import sys, json, csv, collections, re

ST_EXCLUDE = {"xp","kickoff","kick","punt","return","kneel","spike"}

def keep(p):
    # match analytics rules: explicit excludes + ST; allow unknown/empty phase
    if p.get("exclude_from_analytics"):
        return False, ["exclude_flag"]
    st = (p.get("st_fix") or p.get("st") or "").strip().lower()
    if st in ST_EXCLUDE:
        return False, [f"st:{st}"]
    if p.get("special_teams"):
        return False, ["special_teams"]
    ph = (p.get("phase") or "").strip().lower()
    if ph in {"dead","deadball","pre","presnap","post","postplay","timeout","halftime","setup"}:
        return False, [f"phase:{ph}"]
    side = p.get("side")
    if side not in {"offense","defense"}:
        return False, [f"side:{side}"]
    return True, []

def rp_flags_of(p):
    return "run" if p.get("is_run") else ("pass" if p.get("is_pass") else "unknown")

def rp_used_of(p):
    # prefer final label, fall back to flags
    rp_final = (p.get("rp") or "").strip().lower()
    if rp_final not in {"run","pass"}:
        rp_final = rp_flags_of(p)
    return rp_final

def guess_index(p, fallback):
    # Prefer explicit index
    if "index" in p and isinstance(p["index"], int):
        return p["index"]
    # Try "...Clip 061" -> 60 (0-based)
    for field in (p.get("title") or "", p.get("src") or ""):
        m = re.search(r'\b[Cc]lip[ _-]*0*(\d+)\b', field)
        if m:
            try: return int(m.group(1)) - 1
            except: pass
    return fallback

def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/opponent_jenks_silver_20250913")
    plays_path = out / "plays.jsonl"
    audit_dir  = out / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    if not plays_path.exists():
        raise SystemExit(f"[error] {plays_path} not found")

    plays = [json.loads(x) for x in plays_path.read_text().splitlines() if x.strip()]

    kept = []
    debug_rows = []
    off_cnt = def_cnt = 0
    for i, p in enumerate(plays):
        ok, reasons = keep(p)
        side = p.get("side") or ""
        rp_used  = rp_used_of(p)
        rp_flags = rp_flags_of(p)
        if rp_used == "run":
            dir_used = (p.get("run_dir") or "unknown").lower()
        elif rp_used == "pass":
            dir_used = (p.get("direction") or "unknown").lower()
        else:
            dir_used = "unknown"

        idx = guess_index(p, i)

        debug_rows.append({
            "index": idx,
            "kept": str(bool(ok)).lower(),
            "exclude_reasons": ";".join(reasons),
            "side": side,
            "rp_used": rp_used,
            "rp_flags": rp_flags,
            "dir_used": dir_used,
            "run_dir": (p.get("run_dir") or ""),
            "direction": (p.get("direction") or ""),
            "phase": (p.get("phase") or ""),
            "st_fix": (p.get("st_fix") or p.get("st") or ""),
            "exclude_flag": str(bool(p.get("exclude_from_analytics"))).lower(),
            "src": (p.get("src") or ""),
            "title": (p.get("title") or ""),
        })

        if ok:
            kept.append((idx, p))
            if side == "offense": off_cnt += 1
            elif side == "defense": def_cnt += 1

    print("[kept counts]")
    print(f"  offense: {off_cnt}")
    print(f"  defense: {def_cnt}")

    # --- Write debug csv
    dbg_path = audit_dir / "audit_kept_debug.csv"
    with dbg_path.open("w", newline="") as f:
        hdr = ["index","kept","exclude_reasons","side","rp_used","rp_flags","dir_used",
               "run_dir","direction","phase","st_fix","exclude_flag","src","title"]
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader(); w.writerows(sorted(debug_rows, key=lambda r: int(r["index"])))
    print(f"[wrote] {dbg_path}")

    # --- Disagreements (training set): rp_used vs rp_flags on kept plays
    dis_rows = []
    for idx, p in kept:
        rp_u = rp_used_of(p)
        rp_f = rp_flags_of(p)
        if rp_u != rp_f:
            dis_rows.append({
                "index": idx,
                "kept": "true",
                "exclude_reasons": "",
                "side": p.get("side") or "",
                "rp_used": rp_u,
                "rp_flags": rp_f,
                "dir_used": (p.get("run_dir") if rp_u=="run" else (p.get("direction") or "unknown")) if rp_u in {"run","pass"} else "unknown",
                "run_dir": (p.get("run_dir") or ""),
                "direction": (p.get("direction") or ""),
                "phase": (p.get("phase") or ""),
                "st_fix": (p.get("st_fix") or p.get("st") or ""),
                "exclude_flag": "false",
                "src": (p.get("src") or ""),
                "title": (p.get("title") or ""),
            })
    dis_path = audit_dir / "audit_disagreements.csv"
    with dis_path.open("w", newline="") as f:
        hdr = ["index","kept","exclude_reasons","side","rp_used","rp_flags","dir_used",
               "run_dir","direction","phase","st_fix","exclude_flag","src","title"]
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader(); w.writerows(sorted(dis_rows, key=lambda r: int(r["index"])))
    print(f"[wrote] {dis_path}")

    # --- Summary: mirror your quick_* CSVs exactly
    sum_path = audit_dir / "audit_summary.csv"
    rows = []
    for quick in (out/"quick_tendencies_offense.csv", out/"quick_tendencies_defense.csv"):
        if quick.exists():
            rows.extend(list(csv.DictReader(quick.open())))
    with sum_path.open("w", newline="") as f:
        hdr = ["side","bucket","value","count"]
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow({
                "side": r.get("side",""),
                "bucket": r.get("bucket",""),
                "value": r.get("value",""),
                "count": r.get("count","0"),
            })
    print(f"[wrote] {sum_path}")

if __name__ == "__main__":
    main()
