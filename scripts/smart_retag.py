#!/usr/bin/env python3
import argparse, csv, json, subprocess
from pathlib import Path
from tools.json_io import iter_jsonl_safe

POSITIVE_WORDS = {
  "generic": {"WIN","SUCCESS","POSITIVE","TD","TOUCHDOWN","INT","INTERCEPTION","PICK","TAKEAWAY",
              "FF","FORCED FUMBLE","FR","FUMBLE RECOVERY","SACK","TFL","PBU","PASS BREAKUP",
              "EXPLOSIVE","BIG GAIN","FIRST DOWN","STOP","4TH DOWN STOP"},
  "offense": {"TD","TOUCHDOWN","EXPLOSIVE","BIG GAIN","FIRST DOWN","SUCCESS"},
  "defense": {"SACK","TFL","INT","INTERCEPTION","PBU","TAKEAWAY","STOP","4TH DOWN STOP","FF","FR"}
}
NEGATIVE_WORDS = {"CORRECTION","ERROR","MISSED","BUST","FLAG","PENALTY","TURNOVER","ALLOW","ALLOWED"}

def read_jsonl(p: Path):
    return list(iter_jsonl_safe(p))

def fnum(x):
    try: return float(x)
    except: return None

def first_num(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            v=fnum(d[k]); 
            if v is not None: return v
    return None

def first_str(d, keys):
    for k in keys:
        if k in d and d[k]:
            return str(d[k])
    return ""

def overlap(a0,a1,b0,b1, tol=1.0):
    return abs(a0-b0) <= tol and abs(a1-b1) <= tol

def tag_from_grades(grades, s, e, our_team):
    ours_pos=0; ours_neg=0; opp_pos=0
    for g in grades:
        t = first_num(g, ["t","time","ts","timestamp"])
        if t is not None and (t < s or t > e): 
            continue
        team = first_str(g, ["team","side","color","squad"]).upper()
        label= first_str(g, ["label","tag","outcome","note"]).upper()
        # numeric grade can also be a hint
        val  = first_num(g, ["grade","score","value"])

        # keyword buckets
        has_pos = any(w in label for w in POSITIVE_WORDS["generic"])
        has_neg = any(w in label for w in NEGATIVE_WORDS)

        if team == our_team:
            if has_neg or (val is not None and val < 0): ours_neg += 1
            if has_pos or (val is not None and val > 0): ours_pos += 1
        elif team and team != our_team:
            if has_pos or (val is not None and val > 0): opp_pos += 1

    if ours_neg and not ours_pos: return "CORRECTION"
    if ours_pos > opp_pos:        return "WIN"
    return None

def tag_from_predictions(preds, s, e, our_team):
    for p in preds:
        ps = first_num(p, ["start_s","start","t0"])
        pe = first_num(p, ["end_s","end","t1"])
        if ps is None or pe is None: 
            continue
        if not overlap(ps,pe,s,e, tol=1.0):
            continue
        # direct boolean fields
        for k in ("success","our_success","our_win","our_advantage","is_win"):
            if k in p:
                try:
                    return "WIN" if bool(p[k]) else None
                except: pass
        # team outcome fields
        team = first_str(p, ["team","side","color"]).upper()
        label= first_str(p, ["label","tag","outcome"]).upper()
        if team == our_team and label in {"WIN","SUCCESS"}:
            return "WIN"
        if team == our_team and any(w in label for w in NEGATIVE_WORDS):
            return "CORRECTION"
        # numeric advantage
        adv = first_num(p, ["our_advantage","score_delta","yard_delta","success_probability","win_probability"])
        if adv is not None and adv > 0:
            return "WIN"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--our-team", default="WHITE")
    ap.add_argument("--role", choices=["offense","defense","auto"], default="auto",
                    help="Used only for keyword biasing; auto is fine")
    ap.add_argument("--min-win", type=int, default=1, help="min positive signals to call WIN when ambiguous")
    args = ap.parse_args()

    out = Path(args.out)
    plays = read_jsonl(out/"plays.jsonl")
    grades = read_jsonl(out/"grades.jsonl")
    preds  = read_jsonl(out/"play_predictions.jsonl")

    # prepare parts (already cut to clips/tmp/clip_###.mp4)
    tmp = out/"clips"/"tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    tagged=[]
    for idx,d in enumerate(plays, 1):
        s = first_num(d, ["start_s","start","t0","begin"])
        e = first_num(d, ["end_s","end","t1","finish"])
        if s is None or e is None or e <= s: 
            continue
        tag = (tag_from_grades(grades, s, e, args.our_team) 
               or tag_from_predictions(preds, s, e, args.our_team))
        if not tag:
            tag = "NEUTRAL"
        tagged.append((idx,s,e,tag))

    # write timeline
    (out/"dashboards").mkdir(parents=True, exist_ok=True)
    with (out/"dashboards"/"timeline.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["idx","start","end","duration","tag","player","note"])
        for i,s,e,tag in sorted(tagged, key=lambda x:x[1]):
            w.writerow([i,f"{s:.3f}",f"{e:.3f}",f"{(e-s):.3f}",tag,"",""])

    # build win-only highlights
    win_parts=[]
    for i,_,_,tag in tagged:
        p = tmp/f"clip_{i:03d}.mp4"
        if tag=="WIN" and p.exists():
            win_parts.append(p.resolve().as_posix())

    highlights = out/"clips"/"highlights"/"team_highlights.mp4"
    highlights.parent.mkdir(parents=True, exist_ok=True)

    if win_parts:
        lst = out/"clips"/"win_inputs.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in win_parts), encoding="utf-8")
        subprocess.run(["ffmpeg","-hide_banner","-y",
                        "-f","concat","-safe","0","-i",str(lst),
                        "-c:v","libx264","-pix_fmt","yuv420p",
                        "-c:a","aac","-b:a","128k","-movflags","+faststart",
                        str(highlights)], check=True)
        print(f"✅ Highlights rebuilt from WINs: {highlights}")
    else:
        print("⚠️ No WINs found. Leaving existing highlights as-is (all segments).")

    print(f"Wrote timeline: {(out/'dashboards'/'timeline.csv')}")
