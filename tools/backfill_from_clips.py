from __future__ import annotations
import csv, json, subprocess
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
    clips_root = run_dir / "clips"
    if not clips_root.exists():
        print(f"[skip] no clips dir: {clips_root}")
        return 0

    play_dirs = sorted(p for p in clips_root.rglob("PLAY_*") if p.is_dir())
    if not play_dirs:
        print(f"[skip] no per-play folders under {clips_root}")
        return 0

    # ensure targets
    (run_dir / "plays").mkdir(parents=True, exist_ok=True)
    plays_jsonl = run_dir / "plays.jsonl"
    plays_index = run_dir / "plays_index.csv"

    rows = []
    count = 0
    with plays_jsonl.open("w", encoding="utf8") as jf, plays_index.open("w", newline="", encoding="utf8") as cf:
        w = csv.writer(cf)
        w.writerow(["play_id","t0","t1","snap","whistle","clip_path","formation","play_family","outcome","clip_duration"])
        for pdir in play_dirs:
            clip = next((p for p in pdir.glob("*.mp4")), None)
            if not clip:
                continue
            play_id = pdir.name.replace("PLAY_","")
            clip_dur = ffprobe_duration(clip) or 0.0

            # write jsonl (minimal, t0/t1 unknown)
            obj = {
                "play_id": int(play_id) if play_id.isdigit() else play_id,
                "clip_path": str(clip),
                "t0": None, "t1": None,
                "formation": {"name": None, "confidence": 0.0, "candidates": []},
                "playcall": {"name": None, "confidence": 0.0, "candidates": []},
                "outcome": {"yards": 0, "success": False, "explosive": False, "turnover": False, "penalty": False},
                "cues": {},
                "clip_duration": clip_dur,
            }
            jf.write(json.dumps(obj) + "\n")

            # index row
            w.writerow([obj["play_id"], "", "", "", "", str(clip), "", "", "", f"{clip_dur:.3f}"])

            # per-play scaffold
            target = run_dir / "plays" / f"PLAY_{play_id}"
            target.mkdir(parents=True, exist_ok=True)
            (target / "play.json").write_text(json.dumps({
                "play_id": obj["play_id"],
                "formation": obj["formation"],
                "playcall": obj["playcall"],
                "outcome": obj["outcome"],
                "cues": obj["cues"]
            }, indent=2))

            count += 1

    print(f"[backfill] wrote {count} plays -> {plays_jsonl} and {plays_index}")
    return count

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tools/backfill_from_clips.py <run_dir> [<run_dir> ...]")
        sys.exit(2)
    total = 0
    for rd in sys.argv[1:]:
        total += backfill(Path(rd))
    print(f"[done] total plays backfilled: {total}")
