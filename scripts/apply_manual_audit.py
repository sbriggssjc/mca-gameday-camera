#!/usr/bin/env python3
import sys, json, re, subprocess
from pathlib import Path

def parse_audit(path: Path):
    d = {}
    if not path.exists():
        return d
    for line in path.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith("#"): continue
        m = re.match(r"(\d+)\s*,\s*(.+)", line)
        if not m: continue
        d[int(m.group(1))] = m.group(2).strip().lower()
    return d

def safe_concat_write(items, audits, okset, outpath: Path):
    files=[]
    for i, p in enumerate(items, start=1):
        lab = audits.get(i)
        keep = (lab in okset) if lab is not None else True
        if keep:
            src = str(p.get("src",""))
            # escape single quotes for ffmpeg concat
            src = src.replace("'", r"'\''")
            files.append(f"file '{src}'\n")
    outpath.write_text("".join(files))
    return len(files)

def main():
    if len(sys.argv)<2:
        print("usage: apply_manual_audit.py <OUT_DIR>")
        sys.exit(2)
    out = Path(sys.argv[1]).expanduser()
    plays_path = out/"plays.jsonl"
    if not plays_path.exists():
        raise SystemExit(f"missing {plays_path}")

    plays = [json.loads(l) for l in plays_path.read_text().splitlines() if l.strip()]
    # Determine offense plays (prefer final field if present)
    def side_of(p):
        for k in ("lincoln_side_final","lincoln_side","side"):
            v = p.get(k)
            if isinstance(v,str): return v.lower()
        return ""
    offense = [p for p in plays if side_of(p).startswith("off")]

    passes = [p for p in offense if p.get("is_pass") is True]
    runs   = [p for p in offense if p.get("is_run")  is True]

    aud_pass = parse_audit(out/"audit_passes.txt")
    aud_run  = parse_audit(out/"audit_runs.txt")

    PASS_OK = {"pass","offense pass","qbkeeper pass","pass qbkeeper"}
    RUN_OK  = {"run","offense run","run qbkeeper","qbkeeper run"}

    n_run = safe_concat_write(runs, aud_run, RUN_OK, out/"concat_offense_runs.txt")
    n_pas = safe_concat_write(passes, aud_pass, PASS_OK, out/"concat_offense_passes.txt")
    print(f"[concat] runs={n_run} -> {out/'concat_offense_runs.txt'}")
    print(f"[concat] passes={n_pas} -> {out/'concat_offense_passes.txt'}")

    def ff(infile, outfile):
        if Path(infile).exists() and Path(infile).stat().st_size>0:
            subprocess.run([
                "ffmpeg","-y","-f","concat","-safe","0",
                "-i", infile, "-c","copy", outfile
            ], check=False)
        else:
            print(f"[skip] empty or missing {infile}")

    ff(str(out/"concat_offense_runs.txt"),   str(out/"coach_cut_offense_runs.mp4"))
    ff(str(out/"concat_offense_passes.txt"), str(out/"coach_cut_offense_passes.mp4"))
    print("[done] wrote coach_cut_offense_runs.mp4 and coach_cut_offense_passes.mp4")

if __name__ == "__main__":
    main()
