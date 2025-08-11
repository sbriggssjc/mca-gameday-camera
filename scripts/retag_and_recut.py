#!/usr/bin/env python3
import argparse, csv, json, subprocess
from pathlib import Path

def load_jsonl(p:Path):
    if not p.exists(): return []
    out=[]
    for ln in p.read_text().splitlines():
        ln=ln.strip()
        if not ln: continue
        try: out.append(json.loads(ln))
        except: pass
    return out

def pick_time(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            try: return float(d[k])
            except: pass
    return None

def tag_from_grades(grades, s, e, our_team):
    """Very forgiving: counts 'CORRECTION' on our_team vs opponent, or uses numeric grade if present."""
    ours_err = 0; ours_pos=0; opp_pos=0
    for g in grades:
        # match to window by time if present
        t = pick_time(g, "t","time","ts","timestamp")
        if t is not None and (t < s or t > e): 
            continue
        team = (g.get("team") or g.get("side") or g.get("color") or "").upper()
        lab  = (g.get("label") or g.get("tag") or g.get("outcome") or "").upper()
        val  = g.get("grade") or g.get("score") or g.get("value")
        try: val = float(val)
        except: val = None

        if team == our_team:
            if "CORRECTION" in lab or "ERROR" in lab or "MISS" in lab:
                ours_err += 1
            if val is not None and val > 0: 
                ours_pos += 1
        else:
            # opponent positive signal
            if "WIN" in lab or "SUCCESS" in lab:
                opp_pos += 1
            if val is not None and val > 0 and team:
                opp_pos += 1

    if ours_err >= 1 and ours_pos == 0:
        return "CORRECTION"
    if ours_pos > opp_pos:
        return "WIN"
    return None

def tag_from_preds(preds, s, e, our_team):
    """Look for a boolean 'success' or 'our_advantage' style flag; fall back to None."""
    for p in preds:
        t0 = pick_time(p, "start_s","start","t0")
        t1 = pick_time(p, "end_s","end","t1")
        if t0 is not None and t1 is not None:
            # simple overlap check; many preds align 1:1 with windows
            if abs(t0 - s) < 1.0 and abs(t1 - e) < 1.0:
                # various common fields
                for k in ("success","our_success","our_win","our_advantage","is_win"):
                    if k in p:
                        return "WIN" if bool(p[k]) else None
                # if model says 'WIN'/'CORRECTION' outright
                lab=(p.get("label") or p.get("tag") or "").upper()
                if lab in ("WIN","CORRECTION"): return lab
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--our-team", default="WHITE")
    args = ap.parse_args()

    out = Path(args.out)
    plays = load_jsonl(out/"plays.jsonl")
    grades = load_jsonl(out/"grades.jsonl")
    preds  = load_jsonl(out/"play_predictions.jsonl")

    # tag each window
    tagged=[]
    for idx,d in enumerate(plays, 1):
        s = pick_time(d, "start_s","start","t0","begin")
        e = pick_time(d, "end_s","end","t1","finish")
        if s is None or e is None or e <= s: 
            continue
        tag = tag_from_grades(grades, s, e, args.our_team) or tag_from_preds(preds, s, e, args.our_team)
        if not tag: tag = "NEUTRAL"
        tagged.append((idx, s, e, tag))

    # write timeline.csv (sorted by start)
    tagged.sort(key=lambda x: x[1])
    dash = out/"dashboards"; dash.mkdir(parents=True, exist_ok=True)
    with (dash/"timeline.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["idx","start","end","duration","tag","player","note"])
        for i,s,e,tag in tagged:
            w.writerow([i, f"{s:.3f}", f"{e:.3f}", f"{(e-s):.3f}", tag, "", ""])

    # build file list of clips we already cut in clips/tmp
    tmp = out/"clips"/"tmp"
    win_parts = []
    for i,s,e,tag in tagged:
        p = tmp/f"clip_{i:03d}.mp4"
        if tag == "WIN" and p.exists():
            win_parts.append(p.resolve().as_posix())

    # If we have WINs, stitch team_highlights from WIN parts only
    if win_parts:
        lst = out/"clips"/"win_inputs.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in win_parts), encoding="utf-8")
        highlights = out/"clips"/"highlights"/"team_highlights.mp4"
        highlights.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg","-hide_banner","-y","-f","concat","-safe","0","-i",str(lst),
               "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-b:a","128k","-movflags","+faststart", str(highlights)]
        subprocess.run(cmd, check=True)
        print(f"✅ Rebuilt highlights from WINs: {highlights}")
    else:
        print("⚠️ No WIN-tagged parts found; leaving existing highlight as-is.")

if __name__ == "__main__":
    raise SystemExit(main())
