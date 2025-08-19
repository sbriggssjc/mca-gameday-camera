# tools/backfill_from_clips.py
from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path

def ffprobe_duration(mp4: Path) -> float | None:
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration",
             "-of","default=noprint_wrappers=1:nokey=1", str(mp4)],
            stderr=subprocess.STDOUT
        ).decode().strip()
        return float(out) if out else None
    except Exception:
        return None

def backfill(run_dir: Path) -> int:
    run_dir = Path(run_dir)
    clips_root = run_dir / "clips"
    if not clips_root.exists():
        print(f"[skip] no clips under {run_dir}")
        return 0

    # PLAY_XXX/PLAY_XXX.mp4
    pairs: list[tuple[int, Path]] = []
    for play_dir in sorted(clips_root.glob("PLAY_*")):
        pid_str = play_dir.name.split("_")[-1]
        try:
            pid = int(pid_str)
        except Exception:
            continue
        mp4 = play_dir / f"{play_dir.name}.mp4"
        if mp4.exists():
            pairs.append((pid, mp4))

    if not pairs:
        print(f"[skip] no per-play folders under {clips_root}")
        return 0

    plays_jsonl = run_dir / "plays.jsonl"
    plays_csv   = run_dir / "plays_index.csv"
    total = 0

    with plays_jsonl.open("w", encoding="utf8") as jf, \
         plays_csv.open("w", newline="", encoding="utf8") as cf:
        w = csv.writer(cf)
        w.writerow(["play_id","t0","t1","snap","whistle",
                    "clip_path","formation","play_family","outcome","clip_duration"])
        for pid, mp4 in pairs:
            dur = ffprobe_duration(mp4)
            row = {
                "play_id": pid,
                "clip_path": str(mp4),
                "t0": None,
                "t1": None,
                "formation": {"name": None, "confidence": 0.0, "candidates": []},
                "playcall": {"name": None, "confidence": 0.0, "candidates": []},
                "outcome":  {"yards": 0, "success": False, "explosive": False,
                             "turnover": False, "penalty": False},
                "cues": {},
                "clip_duration": round(dur, 3) if dur is not None else None,
            }
            jf.write(json.dumps(row) + "\n")
            w.writerow([pid, "", "", "", "", str(mp4), "", "", "", row["clip_duration"]])
            total += 1

    print(f"[backfill] wrote {total} plays -> {plays_jsonl} and {plays_csv}")
    return total

def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python tools/backfill_from_clips.py <run_dir> [<run_dir> ...]")
        return 2
    total = 0
    for rd in argv[1:]:
        total += backfill(Path(rd))
    print(f"[done] total plays backfilled: {total}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

