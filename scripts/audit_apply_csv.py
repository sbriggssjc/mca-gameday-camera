#!/usr/bin/env python3
import csv, json, sys, re
from pathlib import Path
from collections import Counter, defaultdict

OUT = Path(sys.argv[1]).resolve()
CSVIN = Path(sys.argv[2]).resolve() if len(sys.argv)>2 else OUT/"audit"/"audit_template.csv"
PL  = OUT/"plays.jsonl"
BAK = OUT/"plays.audit_backup.jsonl"

ALLOWED_SIDE = {"offense","defense","unknown",""}
ALLOWED_RP   = {"run","pass","unknown",""}
ALLOWED_DIR  = {"left","right","unknown",""}
ALLOWED_ST   = {"xp","kickoff","kick","punt","return",""}
YESNO = {"y","n",""}

def read_jsonl(p):
    rows=[]
    for ln in p.read_text().splitlines():
        ln=ln.strip()
        if ln:
            try: rows.append(json.loads(ln))
            except: pass
    return rows

def write_jsonl(p, rows):
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)+"\n")

def side_of(p):
    for k in ("side","lincoln_side_final","lincoln_side","lincoln_side_smoothed"):
        v=str(p.get(k,"")).lower()
        if v in ("offense","defense"): return v
    return "unknown"

def rp_of(p):
    if p.get("is_run"): return "run"
    if p.get("is_pass"): return "pass"
    return (str(p.get("rp","unknown")) or "unknown").lower()

def dir_of(p):
    for k in ("dir","direction","run_dir"):
        v=str(p.get(k,"")).lower()
        if v in ("left","right","unknown"): return v
    return "unknown"

def is_special(p):
    v=str(p.get("special_teams_type","")).lower()
    return v in {"xp","extra point","kick","kickoff","punt","return"}

def success_of(p):
    try:
        down = int(p.get("down")) if p.get("down") is not None else None
        dist = float(p.get("distance")) if p.get("distance") is not None else None
        gain = float(p.get("gained_yards")) if p.get("gained_yards") is not None else None
    except: return None
    if down is None or dist is None or gain is None or dist<=0: return None
    thresh = 0.5*dist if down==1 else (0.7*dist if down==2 else 1.0*dist)
    return 1.0 if gain >= thresh else 0.0

def apply_row(p, r):
    # corrections
    sf = str(r.get("side_fix","")).lower()
    rpf= str(r.get("rp_fix","")).lower()
    df = str(r.get("dir_fix","")).lower()
    stf= str(r.get("st_fix","")).lower()
    ex = str(r.get("exclude","")).lower()
    dnf= r.get("down_fix","")
    dif= r.get("distance_fix","")
    gyf= r.get("gained_yards_fix","")
    nf = r.get("notes_fix","")

    errs=[]
    if sf not in ALLOWED_SIDE: errs.append(f"side_fix={sf}")
    if rpf not in ALLOWED_RP: errs.append(f"rp_fix={rpf}")
    if df not in ALLOWED_DIR: errs.append(f"dir_fix={df}")
    if stf not in ALLOWED_ST: errs.append(f"st_fix={stf}")
    if ex not in YESNO: errs.append(f"exclude={ex}")
    if errs: raise ValueError("invalid: "+", ".join(errs))

    if sf in ("offense","defense"): p["side"]=sf
    if rpf in ("run","pass","unknown"):
        p["rp"]=rpf
        p["is_run"]= (rpf=="run")
        p["is_pass"]= (rpf=="pass")
        if rpf=="unknown":
            p.pop("is_run", None); p.pop("is_pass", None)
    if df in ("left","right","unknown"): p["dir"]=df
    if stf in ALLOWED_ST:
        if stf:
            p["special_teams"]=True
            p["special_teams_type"]=stf
        else:
            for k in ("special_teams","special_teams_type"): p.pop(k, None)
    if dnf!="": 
        try: p["down"]=int(dnf)
        except: pass
    if dif!="":
        try: p["distance"]=float(dif)
        except: pass
    if gyf!="":
        try: p["gained_yards"]=float(gyf)
        except: pass
    if nf: p["notes"]= (str(p.get("notes",""))+" | "+str(nf)).strip(" |")
    if ex=="y": p["exclude_from_analytics"]=True
    if ex=="n": p.pop("exclude_from_analytics", None)

def agg_quick(rows, side):
    from collections import Counter
    by=Counter()
    for p in rows:
        rp = p.get("rp","unknown")
        by[("rp", rp)]+=1
        d  = p.get("dir","unknown")
        by[("rp_dir", f"{rp}:{d}")]+=1
    out=[]
    for (b,v),c in by.items():
        out.append({"side":side,"bucket":b,"value":v,"count":c})
    return out

def agg_yards(rows, side):
    from collections import defaultdict
    grp=defaultdict(list)
    for p in rows:
        rp=p.get("rp","unknown"); d=p.get("dir","unknown")
        grp[("rp",rp)].append(p); grp[("rp_dir",f"{rp}:{d}")].append(p)
    out=[]
    for (m,val),g in grp.items():
        gains=[]; succ=[]
        for x in g:
            try: gains.append(float(x.get("gained_yards",0.0)))
            except: gains.append(0.0)
            s=success_of(x)
            if s is not None: succ.append(s)
        n=len(g); avg=sum(gains)/n if n else 0.0
        med=sorted(gains)[n//2] if n else 0.0
        sr =sum(succ)/len(succ) if succ else 0.0
        expl=sum(1 for z in gains if z>=10)/n if n else 0.0
        out.append({"side":side,"metric":m,"value":val,"count":n,
                    "avg_gained":round(avg,3),"median_gained":round(med,3),
                    "success_rate":round(sr,3),"explosive_rate":round(expl,3)})
    return out

def write_csv(path, rows, header):
    with path.open("w", newline="") as f:
        import csv
        w=csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)

def main():
    if not PL.exists(): sys.exit(f"[err] missing {PL}")
    plays=read_jsonl(PL)
    by_idx={int(p.get("index", -1)):p for p in plays}

    # backup
    write_jsonl(BAK, plays)

    # read audit
    with CSVIN.open() as f:
        r=csv.DictReader(f)
        for row in r:
            idx=row.get("index","").strip()
            if not idx.isdigit(): continue
            idx=int(idx)
            p=by_idx.get(idx)
            if not p: 
                print(f"[warn] index {idx} not found; skipping")
                continue
            apply_row(p, row)

    # persist updated plays
    plays_sorted=sorted(by_idx.values(), key=lambda x:int(x.get("index",0)))
    write_jsonl(PL, plays_sorted)
    print(f"[updated] plays.jsonl (backup at {BAK})")

    # filter for analytics (exclude ST and user-excluded)
    def keep_for_analytics(p):
        if p.get("exclude_from_analytics"): return False
        if is_special(p): return False
        return True

    off=[p for p in plays_sorted if str(p.get("side","")).lower()=="offense" and keep_for_analytics(p)]
    deff=[p for p in plays_sorted if str(p.get("side","")).lower()=="defense" and keep_for_analytics(p)]

    # ensure rp/dir defaults
    for p in off+deff:
        if "rp" not in p: p["rp"]=rp_of(p)
        if "dir" not in p: p["dir"]=dir_of(p)

    # write analytics
    qo=agg_quick(off, "offense"); qd=agg_quick(deff,"defense")
    yo=agg_yards(off, "offense"); yd=agg_yards(deff,"defense")

    write_csv(OUT/"quick_tendencies_offense.csv", qo, ["side","bucket","value","count"])
    write_csv(OUT/"quick_tendencies_defense.csv", qd, ["side","bucket","value","count"])
    write_csv(OUT/"yards_tendencies_offense.csv", yo, ["side","metric","value","count","avg_gained","median_gained","success_rate","explosive_rate"])
    write_csv(OUT/"yards_tendencies_defense.csv", yd, ["side","metric","value","count","avg_gained","median_gained","success_rate","explosive_rate"])

    # brief MD
    def split_str(q, side):
        rp=[x for x in q if x["side"]==side and x["bucket"]=="rp"]
        d={x["value"]:x["count"] for x in rp}
        return f"run {d.get('run',0)}, pass {d.get('pass',0)}"
    def top_dirs(q, side):
        rows=[x for x in q if x["side"]==side and x["bucket"]=="rp_dir"]
        rows.sort(key=lambda x:x["count"], reverse=True)
        return "; ".join(f"{x['value']} ({x['count']})" for x in rows[:3]) or "—"

    md=[]
    md.append("## Opponent Offense (special teams & excluded plays removed)")
    md.append(f"- Run/Pass split: {split_str(qo,'offense')}")
    md.append(f"- Top directions: {top_dirs(qo,'offense')}\n")
    md.append("## Opponent Defense (what offenses did vs them; special teams & excluded removed)")
    md.append(f"- Run/Pass split: {split_str(qd,'defense')}")
    md.append(f"- Top directions: {top_dirs(qd,'defense')}\n")
    md.append("## Notes")
    md.append("- Built from audited CSV; special teams and rows with exclude=y are not counted.")
    (OUT/"analysis_report.md").write_text("\n".join(md)+"\n")
    print("[ok] analytics rebuilt (special teams excluded).")

if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: audit_apply_csv.py <OUT_DIR> [audit_csv]")
        sys.exit(2)
    main()
