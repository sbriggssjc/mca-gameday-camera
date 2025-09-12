from __future__ import annotations
import json, pathlib, sys

HELP = """Usage:
python -m analysis.seed_labels OUT --offense 7,9,15 --defense 3,4 --special 20,21
Indexes refer to the order in plays.jsonl (1-based), or pass filenames with --files.
"""

def main():
    out_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    out = pathlib.Path(out_arg) if out_arg else pathlib.Path("output")

    plays_path = out / "plays.jsonl"
    if not plays_path.exists():
        raise SystemExit(
            f"[seed] plays.jsonl not found at '{plays_path}'. "
            "Verify OUT (e.g., OUT=output/opponent_lincoln_20250912). "
            'Use: --files --offense "Wide - Clip 007.mp4"'
        )

    args=sys.argv[2:]
    idx_off, idx_def, idx_sp = [], [], []
    files_mode=False

    def parse_list(s):
        return [x.strip() for x in s.split(",") if x.strip()]

    i=0
    while i<len(args):
        if args[i]=="--offense":
            idx_off+=parse_list(args[i+1]); i+=2
        elif args[i]=="--defense":
            idx_def+=parse_list(args[i+1]); i+=2
        elif args[i]=="--special":
            idx_sp+=parse_list(args[i+1]); i+=2
        elif args[i]=="--files":
            files_mode=True; i+=1
        else:
            print(HELP); return

    rows=[json.loads(x) for x in plays_path.read_text().splitlines() if x.strip()]
    srcs=[r["src"] for r in rows]
    labels={}
    if files_mode:
        # items are filenames
        def to_src(name):
            for s in srcs:
                if pathlib.Path(s).name==name: return s
            return None
        for n in idx_off:
            s=to_src(n); 
            if s: labels[s]="offense"
        for n in idx_def:
            s=to_src(n); 
            if s: labels[s]="defense"
        for n in idx_sp:
            s=to_src(n); 
            if s: labels[s]="special_teams"
    else:
        # items are indices 1-based
        for n in idx_off:
            j=int(n)-1
            if 0<=j<len(srcs): labels[srcs[j]]="offense"
        for n in idx_def:
            j=int(n)-1
            if 0<=j<len(srcs): labels[srcs[j]]="defense"
        for n in idx_sp:
            j=int(n)-1
            if 0<=j<len(srcs): labels[srcs[j]]="special_teams"

    (out/"seed_labels.json").write_text(json.dumps(labels, indent=2))
    print("[seed] wrote", out/"seed_labels.json")

if __name__=="__main__":
    main()
