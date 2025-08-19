# tools/backfill_from_clips.py
from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path
from typing import Any, Dict


def _as_name(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return (x.get("name") or x.get("id") or "").strip()
    return ""


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


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

    # load predictions from pipeline
    pred_map: dict[int, dict] = {}
    pred_file = run_dir / "play_predictions.jsonl"
    if pred_file.exists():
        for line in pred_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            pid_raw = row.get("play_id")
            try:
                pid_key = int(str(pid_raw).split("_")[-1])
            except Exception:
                continue
            pred_map[pid_key] = row

    # existing index to preserve timings
    existing_index: dict[int, dict] = {}
    index_src = run_dir / "plays_index.csv"
    if index_src.exists():
        with index_src.open("r", newline="", encoding="utf8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    pid_key = int(str(r.get("play_id", "")).split("_")[-1])
                except Exception:
                    continue
                existing_index[pid_key] = r

    plays_jsonl = run_dir / "plays.jsonl"
    plays_csv = run_dir / "plays_index.csv"
    total = 0

    with plays_jsonl.open("w", encoding="utf8") as jf, plays_csv.open(
        "w", newline="", encoding="utf8"
    ) as cf:
        fieldnames = [
            "play_id",
            "t0",
            "t1",
            "snap",
            "whistle",
            "clip_path",
            "formation",
            "formation_confidence",
            "play_family",
            "playcall_confidence",
            "outcome",
            "clip_duration",
        ]
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        for pid, mp4 in pairs:
            dur = ffprobe_duration(mp4)
            pred = pred_map.get(pid, {})
            existing = existing_index.get(pid, {})
            formation = pred.get(
                "formation", {"name": None, "confidence": 0.0, "candidates": []}
            )
            playcall = pred.get(
                "playcall", {"name": None, "confidence": 0.0, "candidates": []}
            )
            play_family = pred.get("play_family", "")

            row_json = {
                "play_id": pid,
                "clip_path": str(mp4),
                "t0": existing.get("t0"),
                "t1": existing.get("t1"),
                "formation": formation,
                "playcall": playcall,
                "outcome": {
                    "yards": 0,
                    "success": False,
                    "explosive": False,
                    "turnover": False,
                    "penalty": False,
                },
                "cues": {},
                "clip_duration": round(dur, 3) if dur is not None else None,
            }
            jf.write(json.dumps(row_json) + "\n")

            formation_name = _as_name(formation)
            playcall_name = _as_name(playcall)
            formation_conf = _as_float(formation.get("confidence") if isinstance(formation, dict) else 0.0)
            playcall_conf = _as_float(playcall.get("confidence") if isinstance(playcall, dict) else 0.0)

            w.writerow(
                {
                    "play_id": pid,
                    "t0": existing.get("t0", ""),
                    "t1": existing.get("t1", ""),
                    "snap": existing.get("snap", ""),
                    "whistle": existing.get("whistle", ""),
                    "clip_path": str(mp4),
                    "formation": formation_name,
                    "formation_confidence": formation_conf,
                    "play_family": playcall_name or "",
                    "playcall_confidence": playcall_conf,
                    "outcome": existing.get("outcome", ""),
                    "clip_duration": row_json["clip_duration"],
                }
            )
            print(
                f"[backfill] play {pid}: formation={formation_name} conf={formation_conf} "
                f"playcall={playcall_name} conf={playcall_conf}"
            )
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

