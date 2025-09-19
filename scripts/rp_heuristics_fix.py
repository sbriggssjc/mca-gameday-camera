from pathlib import Path
import json, csv, re, sys
OUT = Path(sys.argv[1] if len(sys.argv)>1 else "output/opponent_jenks_silver_20250913")
plays_path = OUT/"plays.jsonl"
bak = OUT/"plays.autotag_backup.jsonl"

def norm(s):
    return (s or "").strip().lower()

def guess_rp(p):
    # 1) direct fields from pipeline steps
    run_dir = norm(p.get("run_dir"))           # 'left','right','middle','unknown'
    pass_family = norm(p.get("pass_family"))   # 'dropback','screen','rollout','unknown',...
    family = norm(p.get("family"))             # sometimes 'run' or 'pass'
    title = norm(p.get("title"))
    # 2) rules (ordered)
    if run_dir in {"left","right","middle"}:
        return "run", run_dir
    if pass_family and pass_family not in {"", "unknown", "na", "none"}:
        # try to pick a side from 'direction' if present
        d = norm(p.get("direction"))
        d = d if d in {"left","right"} else "unknown"
        return "pass", d
    if family in {"run","pass"}:
        if family == "run":
            d = norm(p.get("run_dir"))
            d = d if d in {"left","right","middle"} else "unknown"
            return "run", d
        else:
            d = norm(p.get("direction"))
            d = d if d in {"left","right"} else "unknown"
            return "pass", d
    # 3) keyword sniffing from title/notes as a last resort
    txt = " ".join([title, norm(p.get("notes"))])
    if re.search(r"\b(counter|power|iso|trap|dive|toss|sweep|draw|qb run|jet)\b", txt):
        return "run", "unknown"
    if re.search(r"\b(pass|drop|throw|screen|rpo)\b", txt):
        return "pass", "unknown"
    return "unknown", "unknown"

def success(gained, down, distance, rp):
    try:
        d, y = int(down or 0), float(gained or 0.0)
        togo = float(distance or 0.0)
    except: return None
    if togo <= 0: return True
    # standard success model
    req = {1: 0.5*togo, 2: 0.7*togo}.get(d, togo)
    return y >= req

def explosive(gained, rp):
    try: y = float(gained or 0.0)
    except: return False
    # simple thresholds
    return y >= (10 if rp=="run" else 15)

# --- read, back up, re-tag ---
lines = [x for x in plays_path.read_text().splitlines() if x.strip()]
plays = [json.loads(x) for x in lines]
bak.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in plays) + "\n")

for p in plays:
    rp, rp_dir = guess_rp(p)
    p["is_run"]  = (rp=="run")
    p["is_pass"] = (rp=="pass")
    p["rp"]      = rp
    p["rp_dir"]  = rp + (":" + rp_dir if rp_dir and rp_dir!="unknown" else ":unknown")

# write updated plays.jsonl
plays_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in plays) + "\n")

# quick_tendencies.csv
from collections import Counter
VALID_SIDES = {"offense", "defense"}

def _resolve_side(play):
    js = str(play.get("jenks_side") or "").strip().lower()
    if js:
        return js if js in VALID_SIDES else ""
    for key in ("lincoln_side_final", "lincoln_side", "side"):
        val = str(play.get(key) or "").strip().lower()
        if val in VALID_SIDES:
            return val
    return ""

off = [p for p in plays if _resolve_side(p) == "offense"]
cnt_rp = Counter(p.get("rp","unknown") for p in off)
cnt_rpdir = Counter(p.get("rp_dir","unknown") for p in off)

qt = OUT/"quick_tendencies.csv"
with qt.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["bucket","value","count"])
    w.writerow(["rp","run", cnt_rp.get("run",0)])
    w.writerow(["rp","pass",cnt_rp.get("pass",0)])
    for k,v in sorted(cnt_rpdir.items()):
        if k.startswith("pass:") or k.startswith("run:"):
            w.writerow(["rp_dir", k, v])

# yards_tendencies.csv (per-bucket)
yt = OUT/"yards_tendencies.csv"
rows=[]
def addrow(metric, value, subset):
    if not subset: return
    gains=[float(p.get("gained_yards") or 0.0) for p in subset]
    succ=[success(p.get("gained_yards"), p.get("down"), p.get("distance"), p.get("rp")) for p in subset]
    succ=[s for s in succ if s is not None]
    exp=[explosive(p.get("gained_yards"), p.get("rp")) for p in subset]
    import statistics as S
    rows.append({
        "metric":metric, "value":value, "count":len(subset),
        "avg_gained": (sum(gains)/len(gains)) if gains else 0.0,
        "median_gained": (S.median(gains) if gains else 0.0),
        "success_rate": (sum(succ)/len(succ) if succ else 0.0),
        "explosive_rate": (sum(exp)/len(exp) if exp else 0.0),
    })

addrow("rp","run", [p for p in off if p.get("is_run")])
addrow("rp","pass",[p for p in off if p.get("is_pass")])
for side in ["left","right","middle","unknown"]:
    addrow("rp_dir", f"run:{side}",  [p for p in off if p.get("rp_dir")==f"run:{side}"])
for side in ["left","right","unknown"]:
    addrow("rp_dir", f"pass:{side}", [p for p in off if p.get("rp_dir")==f"pass:{side}"])

with yt.open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print("Re-tag complete.")
print("Backup:", bak)
print("Updated:", plays_path)
print("Wrote:", qt, yt)
