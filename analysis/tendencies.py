from __future__ import annotations
import json, csv, math, pathlib, collections, re, argparse
from typing import Dict, Any, List, Tuple

Play = Dict[str, Any]

FAMILIES = {
    "inside_run": ["dive","power","iso","counter","trap"],
    "outside_run": ["sweep","jet","stretch","toss"],
    "pa_boot": ["boot","naked","waggle","flare boot","play action","option"],
    "quick_game": ["stick","quick screen","bubble","smoke","hitch","slant","flood","quick"]
}

def normalize(text:str)->str:
    return (text or "").lower().strip()

norm = normalize

def infer_family(play: Play) -> str:
    # Try explicit labels first, then keywords from clip title/notes
    labels = [normalize(play.get("play_label")), normalize(play.get("family")), normalize(play.get("call"))]
    blob = " ".join(labels + [normalize(play.get("title","")), normalize(play.get("notes",""))])
    for fam, keys in FAMILIES.items():
        if any(k in blob for k in keys):
            return fam
    # fallback: if run/pass bool exists
    if play.get("is_run") is True: return "inside_run"
    if play.get("is_pass") is True: return "quick_game"
    return "unknown"

def infer_direction(play: Play) -> str:
    d = normalize(play.get("direction") or play.get("dir"))
    if d in ("right","r","rt","rit","reo"): return "right"
    if d in ("left","l","lt","lit","leo"): return "left"
    # Heuristic: look for tokens in title/notes
    blob = normalize(play.get("title","")+" "+play.get("notes",""))
    if any(t in blob for t in [" right"," rt"," rit"," reo"]): return "right"
    if any(t in blob for t in [" left"," lt"," lit"," leo"]): return "left"
    return "unknown"

FORM_TOKENS = [
  r"\brit\b", r"\blit\b", r"\breo\b", r"\bleo\b",
  r"\brend\b", r"\blend\b",
  r"\btrips?\b", r"\btwins?\b", r"\bbunch\b", r"\bwing\b", r"\btight\b",
  r"\bpistol\b", r"\bgun\b", r"\bshotgun\b", r"\bi[- ]?formation\b"
]

def infer_form(p: Play) -> str:
    cand = norm(p.get("formation") or p.get("set"))
    if cand: return cand
    blob = norm((p.get("title") or "")+" "+(p.get("notes") or ""))
    for pat in FORM_TOKENS:
        if re.search(pat, blob): return re.sub(r"\\b", "", pat).strip("\\")
    return "unknown"

def load_plays(out_dir: pathlib.Path) -> List[Play]:
    for fname in ["plays.jsonl","detections.jsonl","events.jsonl"]:
        p = out_dir / fname
        if p.exists():
            with p.open() as f:
                return [json.loads(line) for line in f if line.strip()]
    return []

def summarize(plays: List[Play]) -> Dict[str, Any]:
    sums = {
        "total": len(plays),
        "by_family": collections.Counter(),
        "by_formation": collections.Counter(),
        "by_direction": collections.Counter(),
        "run_pass": collections.Counter(),
        "first_down_calls": collections.Counter(),
        "third_and_medium_plus": collections.Counter(),
        "explosives": 0
    }
    for p in plays:
        fam = infer_family(p); sums["by_family"][fam]+=1
        form = infer_form(p); sums["by_formation"][form]+=1
        dirn = infer_direction(p); sums["by_direction"][dirn]+=1
        if p.get("is_run") is True:
            sums["run_pass"]["run"] += 1
        elif p.get("is_pass") is True:
            sums["run_pass"]["pass"] += 1
        else:
            af = p.get("auto_flow") or {}
            mag_med = af.get("mag_med", 0)
            vy_med = af.get("vy_med", 0)
            if mag_med >= 0.03:
                rp_guess = "run" if abs(vy_med) < 0.03 else "pass"
                sums["run_pass"][rp_guess] += 1
            else:
                sums["run_pass"]["unknown"] += 1

        down = p.get("down")
        togo = p.get("distance") or p.get("to_go")
        if down == 1: sums["first_down_calls"][fam]+=1
        try:
            if down == 3 and float(togo or 0) >= 5:
                sums["third_and_medium_plus"][fam]+=1
        except Exception:
            pass

        # explosive heuristic: >=12 yards run or >=16 yards pass if gain present
        gain = p.get("yards") or p.get("gain")
        try:
            g = float(gain)
            if (p.get("is_run") and g >= 12) or (p.get("is_pass") and g >= 16):
                sums["explosives"] += 1
        except Exception:
            pass
        outcome = p.get("auto_outcome")
        if outcome:
            sums.setdefault("outcomes", collections.Counter())
            sums["outcomes"][outcome] += 1
    return sums

def write_csv(out_dir: pathlib.Path, plays: List[Play], sums: Dict[str, Any], suffix: str = ""):
    csv_path = out_dir / f"tendencies{suffix}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric","key","value"])
        w.writerow(["total","plays",sums["total"]])
        for k,v in sums["run_pass"].items(): w.writerow(["run_pass",k,v])
        for k,v in sums["by_family"].items(): w.writerow(["family",k,v])
        for k,v in sums["by_formation"].items(): w.writerow(["formation",k,v])
        for k,v in sums["by_direction"].items(): w.writerow(["direction",k,v])
        for k,v in sums.get("outcomes", {}).items(): w.writerow(["outcome",k,v])
    return csv_path


def write_md(out_dir: pathlib.Path, sums: Dict[str, Any], suffix: str = ""):
    md = out_dir / f"tendencies{suffix}.md"
    total = max(1, sums["total"])
    def pct(n): return f"{(100.0*n/total):.1f}%"
    def block(counter: collections.Counter, title: str) -> str:
        rows = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
        lines = [f"### {title}"]
        for k,v in rows[:8]:
            lines.append(f"- **{k}**: {v} ({pct(v)})")
        return "\n".join(lines)

    content = f"""# Opponent Tendencies

**Total plays:** {sums['total']}  
**Explosives:** {sums['explosives']}

{block(sums['run_pass'], "Run/Pass")}
{block(sums['by_family'], "Play Families")}
{block(sums['by_formation'], "Formations (top)")}
{block(sums['by_direction'], "Direction")}
{block(sums.get('outcomes', collections.Counter()), "Outcomes")}
### Situational
- **1st down (by family):** {dict(sums['first_down_calls'])}
- **3rd & 5+ (by family):** {dict(sums['third_and_medium_plus'])}
"""
    md.write_text(content)
    return md


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--only-lincoln-offense", action="store_true")
    ap.add_argument("--only-lincoln-defense", action="store_true")
    ap.add_argument("--use-smoothed-side", dest="use_smoothed_side", action="store_true")
    ap.add_argument("--no-use-smoothed-side", dest="use_smoothed_side", action="store_false")
    ap.set_defaults(use_smoothed_side=True)
    ap.add_argument(
        "--exclude-phase",
        default="special_teams,unknown",
        help="comma list of phases to exclude",
    )
    ap.add_argument("--min-side-conf", type=float, default=0.40)
    return ap.parse_args()


def main():
    args = parse_args()
    out = pathlib.Path(args.out_dir)
    plays = load_plays(out)
    excl = set()
    if args.exclude_phase:
        excl = {x.strip() for x in args.exclude_phase.split(",") if x.strip()}
    plays = [p for p in plays if p.get("phase") not in excl]

    side_key = "lincoln_side_final" if args.use_smoothed_side else "lincoln_side"

    if args.min_side_conf:
        plays = [
            p
            for p in plays
            if p.get(side_key) == "unknown"
            or float(p.get("lincoln_side_conf", 0)) >= args.min_side_conf
        ]

    if args.only_lincoln_offense:
        plays = [p for p in plays if p.get(side_key) == "offense"]
    if args.only_lincoln_defense:
        plays = [p for p in plays if p.get(side_key) == "defense"]

    sums = summarize(plays)
    suffix = ""
    if args.only_lincoln_offense:
        suffix += "_offense"
    if args.only_lincoln_defense:
        suffix += "_defense"
    if args.min_side_conf:
        suffix += f"_conf{int(args.min_side_conf*100)}"
    if args.exclude_phase:
        suffix += "_nophase"
    if args.use_smoothed_side:
        suffix += "_smooth"
    write_csv(out, plays, sums, suffix)
    write_md(out, sums, suffix)


if __name__ == "__main__":
    main()
