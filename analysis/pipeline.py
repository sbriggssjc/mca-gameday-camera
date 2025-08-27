from __future__ import annotations
import argparse, json, csv, os, sys, pathlib, subprocess, shlex
from typing import Dict, Any, List

# Import robustly whether run as a module or a script
try:
    from .segmentation import segment_video   # existing
    from .formation_detector import detect_formations  # existing
    from .play_classifier import classify_plays        # we'll ensure this exists
except Exception:
    # fall back if executed as -m analysis.pipeline from repo root without package context
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
    from analysis.segmentation import segment_video
    from analysis.formation_detector import detect_formations
    from analysis.play_classifier import classify_plays

def _ffmpeg(*args: str) -> None:
    cmd = ["ffmpeg", "-y", *args]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in s)

def run_pipeline(
    video: str,
    team: str,
    playbook_path: str,
    out_dir: str,
    min_play_gap: float,
    min_play_length: float,
    generate_report: bool,
    generate_clips: bool,
    highlights: bool = True,
    overlay: bool = False,
) -> str:
    video = os.path.abspath(video)
    out_dir = os.path.abspath(out_dir)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Build run dir
    tag = pathlib.Path(video).stem
    # short hash to avoid collisions
    short = hex(abs(hash((tag, playbook_path, min_play_gap, min_play_length))) & ((1<<44)-1))[2:]
    run_dir = os.path.join(out_dir, "games", f"{_safe_name(tag)}__{short}")
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)

    # Load playbook
    with open(playbook_path, "r") as f:
        playbook = json.load(f)
    print(f"[playbook] source={playbook_path}")
    print(f"[playbook] OK: loaded playbook from {playbook_path}")

    # Segment
    segments = segment_video(video, min_play_gap=min_play_gap, min_play_length=min_play_length)
    print(f"[config] min_play_length={min_play_length} min_play_gap={min_play_gap} "
          f"report={generate_report} clips={generate_clips} highlights={highlights} overlay={overlay}")
    print(f"[pipeline] segments detected: {len(segments)}")

    # Detect formations + classify
    rows: List[Dict[str, Any]] = []
    formations = detect_formations(video, segments)
    classifications = classify_plays(video, segments, formations, playbook)

    # Ensure each row has a sane schema; candidates is a list[str] (possibly empty)
    for idx, seg in enumerate(segments, start=1):
        pid = f"PLAY_{idx:03d}"
        fdet = formations.get(pid, {})
        cdet = classifications.get(pid, {})
        formation = fdet.get("formation", "Unknown")
        fconf = float(fdet.get("confidence", 0.0))
        family = cdet.get("play_family") or cdet.get("label") or "Unknown"
        pconf = float(cdet.get("confidence", 0.0))
        outcome = cdet.get("outcome", "")
        # normalize candidates
        candidates = cdet.get("candidates") or []
        if not isinstance(candidates, list):
            candidates = []
        # clip path (filled after export step)
        clip_path = ""

        rows.append(dict(
            play_id=pid,
            t0=float(seg["t0"]), t1=float(seg["t1"]),
            snap=float(seg.get("snap", seg["t0"])),
            whistle=float(seg.get("whistle", seg["t1"])),
            clip_path=clip_path,
            formation=formation,
            formation_confidence=fconf,
            play_family=family,
            playcall_confidence=pconf,
            outcome=outcome,
            clip_duration=max(0.0, float(seg["t1"]) - float(seg["t0"])),
            candidates=";".join(candidates),  # new column
        ))

    # Write plays_index.csv (header tolerant downstream)
    csv_path = os.path.join(run_dir, "plays_index.csv")
    fieldnames = ["play_id","t0","t1","snap","whistle","clip_path",
                  "formation","formation_confidence","play_family",
                  "playcall_confidence","outcome","clip_duration","candidates"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Export clips deterministically from source video
    clips_root = os.path.join(run_dir, "clips")
    if generate_clips:
        pathlib.Path(clips_root).mkdir(parents=True, exist_ok=True)
        for r in rows:
            pid = r["play_id"]
            pdir = os.path.join(clips_root, pid)
            pathlib.Path(pdir).mkdir(parents=True, exist_ok=True)
            mp4 = os.path.join(pdir, f"{pid}.mp4")
            t0, t1 = float(r["t0"]), float(r["t1"])
            dur = max(0.1, t1 - t0)
            # Use re-encode for broad compatibility (Jetson ffmpeg build OK)
            _ffmpeg("-ss", f"{t0:.3f}", "-i", video, "-t", f"{dur:.3f}",
                    "-an", "-vf", "scale=720:-2:flags=lanczos", mp4)
            r["clip_path"] = mp4

        # Rewrite CSV with clip_path filled
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Minimal JSON report (for Drive)
    if generate_report:
        rep = {
            "video": video,
            "run_dir": run_dir,
            "n_segments": len(segments),
            "generated_clips": bool(generate_clips),
            "plays": rows,
        }
        with open(os.path.join(run_dir, "report.json"), "w") as f:
            json.dump(rep, f, indent=2)

    print(f"[pipeline] run complete -> {run_dir}")
    return run_dir

def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--team", required=True)              # retained for future use
    p.add_argument("--playbook", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-play-gap", type=float, default=1.5)
    p.add_argument("--min-play-length", type=float, default=3.0)
    p.add_argument("--generate-report", action="store_true")
    p.add_argument("--generate-clips", action="store_true")
    args = p.parse_args(argv)

    run_pipeline(
        video=args.video, team=args.team, playbook_path=args.playbook,
        out_dir=args.out, min_play_gap=args["min_play_gap"] if isinstance(args,dict) else args.min_play_gap,
        min_play_length=args["min_play_length"] if isinstance(args,dict) else args.min_play_length,
        generate_report=args.generate_report, generate_clips=args.generate_clips,
    )

if __name__ == "__main__":
    main()
