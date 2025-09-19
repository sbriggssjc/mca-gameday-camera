#!/usr/bin/env python3
import csv
import json
import re
import statistics as stats
from pathlib import Path

from side_utils import side_for

# ----- helpers ---------------------------------------------------------------


def read_jsonl(p: Path):
    rows = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass
    return rows


def write_jsonl(p: Path, rows):
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def safe_lower(x, default="unknown"):
    if x is None:
        return default
    s = str(x).strip().lower()
    return s if s else default


def parse_audit_line(line: str):
    # strip comments
    line = line.split("#", 1)[0].strip()
    if not line:
        return None
    # accept "k: v, k2: v2" style
    parts = [x.strip() for x in re.split(r"\s*,\s*", line) if x.strip()]
    out = {}
    for part in parts:
        if ":" not in part:
            continue
        k, v = [t.strip() for t in part.split(":", 1)]
        out[k.lower()] = v.strip()
    if not out:
        return None
    # coerce
    if "idx" in out:
        digits = re.sub(r"[^\d]", "", out["idx"])
        if digits:
            try:
                out["idx"] = int(digits)
            except Exception:
                pass
        else:
            out.pop("idx")
    return out


clip_num_re = re.compile(r"[Cc]lip[^0-9]*([0-9]{1,3})")


def idx_from_src(src: str):
    m = clip_num_re.search(Path(src).name)
    return int(m.group(1)) if m else None


def get_side(p):
    explicit = safe_lower(p.get("jenks_side"), "")
    if explicit:
        return explicit

    preferred = side_for("jenks", p)
    pref_norm = safe_lower(preferred, "")
    if pref_norm:
        return pref_norm
    for key in ("side", "lincoln_side_final", "lincoln_side", "lincoln_side_smoothed"):
        v = safe_lower(p.get(key), "")
        if v in ("offense", "defense"):
            return v
    return "unknown"


def success_for_down(gained, down, distance):
    # NCAA-ish success: 1st>=50% line-to-gain, 2nd>=70%, 3rd/4th>=100%
    try:
        g = float(gained if gained is not None else 0)
        d = int(down) if down is not None else None
        dist = float(distance) if distance is not None else None
    except Exception:
        return None
    if dist is None or dist <= 0:  # fallback: any positive gain
        return g > 0
    if d == 1:
        req = 0.50 * dist
    elif d == 2:
        req = 0.70 * dist
    else:
        req = 1.00 * dist
    return g >= req


def is_explosive(gained, rp):
    try:
        g = float(gained if gained is not None else 0)
    except Exception:
        return None
    if rp == "run":
        return g >= 10
    if rp == "pass":
        return g >= 15
    return g >= 12


def select_dir(p):
    # pick a direction label consistently
    if safe_lower(p.get("dir"), "") not in ("", "unknown"):
        return safe_lower(p.get("dir"))
    if p.get("is_pass"):
        return safe_lower(p.get("direction"))
    if p.get("is_run"):
        return safe_lower(p.get("run_dir") or p.get("direction"))
    return safe_lower(p.get("direction"))


def num(x):
    try:
        return float(x)
    except Exception:
        return None


SPECIAL_TEAM_KEYWORDS = (
    "xp",
    "extra point",
    "kick",
    "kickoff",
    "return",
    "punt",
)


def detect_special_keyword(value):
    text = safe_lower(value, "")
    if not text or text == "unknown":
        return ""
    for keyword in SPECIAL_TEAM_KEYWORDS:
        if text == keyword or keyword in text:
            return keyword
    return ""


def tag_special(play, audit_info=None):
    # prefer explicit audit flags, but also scan notes for common keywords
    existing_type = play.get("special_teams_type")
    candidates = []
    play.pop("special_teams_flag", None)
    play.pop("special_teams_type", None)
    if audit_info:
        candidates.extend([audit_info.get("st"), audit_info.get("notes")])
    candidates.extend(
        [
            play.get("special_teams"),
            existing_type,
            play.get("notes"),
        ]
    )
    for candidate in candidates:
        keyword = detect_special_keyword(candidate)
        if keyword:
            play["special_teams_flag"] = True
            play["special_teams_type"] = keyword
            # Maintain compatibility with older tooling that expects the raw string
            if not isinstance(play.get("special_teams"), str) or not play.get("special_teams"):
                play["special_teams"] = keyword
            return True
    return False


def rp_of(play, audit_info=None):
    if audit_info:
        rp_val = safe_lower(audit_info.get("rp"), "")
        if rp_val in ("run", "pass", "unknown"):
            return rp_val
    rp_val = safe_lower(play.get("rp"), "")
    if rp_val in ("run", "pass", "unknown"):
        return rp_val
    if play.get("is_run"):
        return "run"
    if play.get("is_pass"):
        return "pass"
    return "unknown"


def dir_of(play, audit_info=None):
    for candidate in (
        safe_lower((audit_info or {}).get("dir"), ""),
        safe_lower(play.get("dir"), ""),
        safe_lower(play.get("direction"), ""),
        safe_lower(play.get("run_dir"), ""),
    ):
        if candidate in ("left", "right", "unknown", "middle", "center"):
            return candidate
    return "unknown"


def is_special(play):
    if play.get("special_teams_flag") is True:
        return True
    special_val = play.get("special_teams")
    if isinstance(special_val, bool):
        if special_val:
            return True
    else:
        if detect_special_keyword(special_val):
            return True
    return detect_special_keyword(play.get("special_teams_type")) != ""

# ----- main -----------------------------------------------------------------
def main(out_dir: str):
    out = Path(out_dir)
    plays_p = out / "plays.jsonl"
    assert plays_p.exists(), f"not found: {plays_p}"
    plays = read_jsonl(plays_p)

    # normalize side using fallbacks and persist it
    for p in plays:
        s = get_side(p)
        if s in ("offense", "defense"):
            p["side"] = s
            p["jenks_side"] = s
            p["metro_side"] = "defense" if s == "offense" else "offense"

    # read audits if present
    audits = []
    for fname in ("audit_runs.txt", "audit_passes.txt"):
        f = out / fname
        if f.exists():
            for ln in f.read_text().splitlines():
                rec = parse_audit_line(ln)
                if rec:
                    audits.append(rec)

    audit_by_idx = {}
    for rec in audits:
        idx = rec.get("idx")
        if isinstance(idx, int):
            audit_by_idx.setdefault(idx, {}).update(rec)

    # build index -> play map (supports either explicit index or clip src)
    idx_to_i = {}
    play_indices = {}
    for i, p in enumerate(plays):
        idx = None
        play_index = p.get("index")
        if isinstance(play_index, int):
            idx = play_index
        elif isinstance(play_index, str) and play_index.isdigit():
            idx = int(play_index)
        if idx is None:
            idx = idx_from_src(p.get("src", ""))
        if idx is not None:
            play_indices[i] = idx
            idx_to_i.setdefault(idx, i)

    # apply audit
    touched = 0
    for a in audits:
        target = None
        idx = a.get("idx") if isinstance(a.get("idx"), int) else None
        if idx is not None:
            i = idx_to_i.get(idx)
            if i is not None:
                target = plays[i]
        # also allow src: substring match (optional)
        if target is None and a.get("src"):
            src_val = str(a["src"])
            for p in plays:
                if src_val.lower() in str(p.get("src", "")).lower():
                    target = p
                    break
        if target is None:
            continue

        # side
        side_val = safe_lower(a.get("side"), "")
        if side_val in ("offense", "defense"):
            target["side"] = side_val
            target["jenks_side"] = side_val
            target["metro_side"] = "defense" if side_val == "offense" else "offense"

        # rp -> is_run/is_pass
        rp = safe_lower(a.get("rp"), "")
        if rp in ("run", "pass", "unknown"):
            target["is_run"] = rp == "run"
            target["is_pass"] = rp == "pass"
            if rp == "unknown":
                target["is_run"] = False
                target["is_pass"] = False

        # direction (optional)
        if "dir" in a and a["dir"]:
            d_raw = a["dir"]
            d = safe_lower(d_raw, "")
            if target.get("is_run"):
                target["run_dir"] = d
            else:
                target["direction"] = d
            target["dir"] = d

        # special teams tag (optional)
        if a.get("st"):
            target["special_teams"] = a["st"]
            keyword = detect_special_keyword(a["st"])
            if keyword:
                target["special_teams_flag"] = True
                target["special_teams_type"] = keyword

        # notes (append)
        if a.get("notes"):
            prev = target.get("notes", "")
            sep = " ; " if prev else ""
            target["notes"] = f"{prev}{sep}{a['notes']}"

        touched += 1

    # ensure plays capture normalized rp/dir and special-teams metadata
    for i, p in enumerate(plays):
        idx = play_indices.get(i)
        kv = audit_by_idx.get(idx, {}) if idx is not None else {}

        rp_val = rp_of(p, kv)
        p["rp"] = rp_val

        dir_val = dir_of(p, kv)
        p["dir"] = dir_val

        tag_special(p, kv)

        # Re-affirm normalized side after audit updates
        s = get_side(p)
        if s in ("offense", "defense"):
            p["side"] = s
            p["jenks_side"] = s
            p["metro_side"] = "defense" if s == "offense" else "offense"

    # write back
    bak = out / "plays.audit_sync_backup.jsonl"
    write_jsonl(bak, plays)
    write_jsonl(plays_p, plays)

    # ---------------- recompute analytics -----------------------------------
    def side_filter(side):
        filtered = []
        for p in plays:
            explicit = safe_lower(p.get("jenks_side"), "")
            if explicit:
                if explicit not in ("offense", "defense") or explicit != side:
                    continue
            else:
                inferred = safe_lower(get_side(p), "")
                if inferred not in ("offense", "defense") or inferred != side:
                    continue
            if is_special(p):
                continue
            filtered.append(p)
        return filtered

    def compile_quick(rows, side_label):
        # rp + rp:dir counts
        cnt = []
        rp_counts = {}
        rpdir_counts = {}
        for p in rows:
            if is_special(p):
                continue
            rp = rp_of(p)
            rp_counts[rp] = rp_counts.get(rp, 0) + 1
            d = dir_of(p)
            rpdir_counts[(rp, d)] = rpdir_counts.get((rp, d), 0) + 1

        out_rows = []
        for rp, c in sorted(rp_counts.items(), key=lambda kv: (kv[0], -kv[1])):
            out_rows.append({"side": side_label, "bucket": "rp", "value": rp, "count": c})
        for (rp, d), c in sorted(rpdir_counts.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            out_rows.append({"side": side_label, "bucket": "rp_dir", "value": f"{rp}:{d}", "count": c})
        return out_rows

    def compile_yards(rows, side_label):
        # group by rp and rp_dir; compute averages, success_rate, explosive_rate
        groups = {}
        for p in rows:
            if is_special(p):
                continue
            rp = rp_of(p)
            d = dir_of(p)
            for key in [("rp", rp), ("rp_dir", f"{rp}:{d}")]:
                groups.setdefault(key, []).append(p)

        def row_metric(metric,value,plays_):
            g = [num(p.get("gained_yards")) for p in plays_ if num(p.get("gained_yards")) is not None]
            avg = sum(g)/len(g) if g else 0.0
            med = (stats.median(g) if g else 0.0)
            succ = []
            expl = []
            for p in plays_:
                s = success_for_down(p.get("gained_yards"), p.get("down"), p.get("distance"))
                if s is not None: succ.append(bool(s))
                e = is_explosive(p.get("gained_yards"), rp_of(p))
                if e is not None: expl.append(bool(e))
            succ_rate = (sum(succ)/len(succ) if succ else 0.0)
            expl_rate = (sum(expl)/len(expl) if expl else 0.0)
            return {
                "side": side_label, "metric": metric, "value": value,
                "count": len(plays_), "avg_gained": round(avg,3),
                "median_gained": round(med,3),
                "success_rate": round(succ_rate,3),
                "explosive_rate": round(expl_rate,3),
            }

        rows_out = []
        for (metric,value), plist in sorted(groups.items()):
            rows_out.append(row_metric(metric,value,plist))
        return rows_out

    off = side_filter("offense")
    de  = side_filter("defense")

    quick_off = compile_quick(off, "offense")
    quick_de  = compile_quick(de,  "defense")
    yards_off = compile_yards(off, "offense")
    yards_de  = compile_yards(de,  "defense")

    def write_csv(path, fieldnames, rows):
        with open(path,"w",newline="") as f:
            w=csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows: w.writerow(r)

    write_csv(out/"quick_tendencies_offense.csv", ["side","bucket","value","count"], quick_off)
    write_csv(out/"quick_tendencies_defense.csv", ["side","bucket","value","count"], quick_de)
    write_csv(out/"yards_tendencies_offense.csv",
              ["side","metric","value","count","avg_gained","median_gained","success_rate","explosive_rate"],
              yards_off)
    write_csv(out/"yards_tendencies_defense.csv",
              ["side","metric","value","count","avg_gained","median_gained","success_rate","explosive_rate"],
              yards_de)

    # also write unified files (handy for spreadsheets)
    write_csv(out/"quick_tendencies.csv", ["side","bucket","value","count"], quick_off+quick_de)
    write_csv(out/"yards_tendencies.csv",
              ["side","metric","value","count","avg_gained","median_gained","success_rate","explosive_rate"],
              yards_off+yards_de)

    # compact markdown summary
    def topn(rows, bucket, n=3):
        filt=[r for r in rows if r["bucket"]==bucket]
        filt.sort(key=lambda r: r["count"], reverse=True)
        return filt[:n]

    md = []
    def add(title, rows_quick, rows_yards, side):
        md.append(f"## {title}")
        # RP split
        rp = [r for r in rows_quick if r["bucket"]=="rp"]
        rp.sort(key=lambda r: {"run":0,"pass":1,"unknown":2}.get(r["value"],3))
        rp_txt = ", ".join(f"{r['value']} {r['count']}" for r in rp)
        md.append(f"- Run/Pass split: {rp_txt}")
        # Direction tendencies (top 3)
        tops = topn(rows_quick, "rp_dir", 3)
        if tops:
            md.append("- Top directions: " + "; ".join(f"{r['value']} ({r['count']})" for r in tops))
        # Efficiency quick take
        def avg_succ(val):
            rs = [y for y in rows_yards if y["metric"]=="rp" and y["value"]==val]
            return rs[0]["success_rate"] if rs else 0.0
        md.append(f"- Success rate: run {avg_succ('run'):.2f} | pass {avg_succ('pass'):.2f}")
        md.append("")
    add("Opponent Offense", quick_off, yards_off, "offense")
    add("Opponent Defense (what offenses did vs them)", quick_de, yards_de, "defense")

    # lightweight heuristics for ideas
    def heuristics():
        lines=[]
        # offense bias
        def get_rate(rows, rp, side):
            for r in rows:
                if r["side"]==side and r.get("metric")=="rp" and r.get("value")==rp:
                    return r["success_rate"]
            return 0.0
        off_run_s = get_rate(yards_off,"run","offense")
        off_pass_s = get_rate(yards_off,"pass","offense")
        if off_run_s > off_pass_s + 0.08:
            lines.append("Offense: lean run—higher success than pass.")
        elif off_pass_s > off_run_s + 0.08:
            lines.append("Offense: lean pass—higher success than run.")
        # defense weakness
        de_run_s = get_rate(yards_de,"run","defense")
        de_pass_s = get_rate(yards_de,"pass","defense")
        if de_run_s > de_pass_s + 0.08:
            lines.append("Defense: appears softer vs RUN (offenses more successful running).")
        elif de_pass_s > de_run_s + 0.08:
            lines.append("Defense: appears softer vs PASS (offenses more successful passing).")
        if not lines:
            lines.append("No glaring bias by split—exploit directional tendencies.")
        return lines

    md.append("## Quick coaching notes")
    for line in heuristics():
        md.append(f"- {line}")
    md.append("")
    md.append("## Notes")
    md.append("- Special teams (XP/kick/kickoff/punt/returns) are excluded from all tables.")

    (out/"analysis_report.md").write_text("\n".join(md) + "\n")

    print(f"[ok] audit synced to analytics (special teams excluded). Updated CSVs + analysis_report.md in {out}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: sync_audit_to_analytics.py <OUTPUT_DIR>")
        raise SystemExit(2)
    main(sys.argv[1])
