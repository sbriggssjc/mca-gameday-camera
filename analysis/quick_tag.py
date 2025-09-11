from __future__ import annotations
import json
import pathlib
import sys

HELP = """
Controls:
  r = run, p = pass, u = unknown
  L = left, R = right, D = unknown direction
  1/2/3/4 sets down; digits 0-9 (after & or alone) set distance (e.g., 1 then 0 -> 10)
  f = family cycle (inside_run -> outside_run -> pa_boot -> quick_game -> unknown)
  n = next play, b = back play
  h = help, q = quit/save
"""

FAMILIES = ["inside_run", "outside_run", "pa_boot", "quick_game", "unknown"]

def load(out_dir: pathlib.Path):
    p = out_dir / "plays.jsonl"
    plays = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return p, plays

def save(path: pathlib.Path, plays):
    with path.open("w") as f:
        for pl in plays:
            f.write(json.dumps(pl, ensure_ascii=False) + "\n")

def main(out_dir_str: str):
    out = pathlib.Path(out_dir_str)
    path, plays = load(out)
    i = 0
    fam_idx = {k: i for i, k in enumerate(FAMILIES)}
    print(HELP)
    while 0 <= i < len(plays):
        pl = plays[i]
        print(f"\n[{i+1}/{len(plays)}] {pl.get('title')}")
        print(
            f"  family={pl.get('family', 'unknown')}  is_run={pl.get('is_run')}  "
            f"is_pass={pl.get('is_pass')}"
        )
        print(
            f"  direction={pl.get('direction', 'unknown')}  down={pl.get('down')}  "
            f"distance={pl.get('distance')}"
        )
        try:
            key = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            key = "q"
        lower = key.lower()
        if lower == "q":
            break
        if lower == "h":
            print(HELP)
            continue
        if lower == "n":
            i += 1
            continue
        if lower == "b":
            i = max(0, i - 1)
            continue
        if lower == "r" and key == "r":
            pl["is_run"] = True
            pl["is_pass"] = False
        elif lower == "p":
            pl["is_pass"] = True
            pl["is_run"] = False
        elif lower == "u":
            pl["is_run"] = None
            pl["is_pass"] = None
        elif key == "L":
            pl["direction"] = "left"
        elif key == "R":
            pl["direction"] = "right"
        elif key == "D":
            pl["direction"] = "unknown"
        elif lower in ["1", "2", "3", "4"]:
            pl["down"] = int(lower)
        elif key.isdigit():
            cur = int(pl.get("distance") or 0)
            pl["distance"] = (cur * 10 + int(key)) if cur < 100 else int(key)
        elif lower == "f":
            cur = pl.get("family") or "unknown"
            nxt = FAMILIES[(fam_idx.get(cur, len(FAMILIES) - 1) + 1) % len(FAMILIES)]
            pl["family"] = nxt
            pl["is_run"] = (
                True
                if nxt in ("inside_run", "outside_run")
                else False
                if nxt in ("pa_boot", "quick_game")
                else None
            )
            pl["is_pass"] = (
                True
                if nxt in ("pa_boot", "quick_game")
                else False
                if nxt in ("inside_run", "outside_run")
                else None
            )
        else:
            print("Unknown key. Press h for help.")
            continue
        plays[i] = pl
    save(path, plays)
    print("Saved.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output")
