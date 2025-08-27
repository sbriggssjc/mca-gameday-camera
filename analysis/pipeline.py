from __future__ import annotations

try:
    # normal, when run via `python -m analysis.pipeline`
    from .play_classifier import classify_plays
    from .segmentation import segment_video
except Exception:
    # fallback when run as script or when PYTHONPATH is set to repo root
    from analysis.play_classifier import classify_plays  # type: ignore
    from analysis.segmentation import segment_video       # type: ignore

import os, json, argparse, pathlib

CSV_HEADER = [
    "play_id","t0","t1","snap","whistle","clip_path",
    "formation","formation_confidence","play_family",
    "playcall_confidence","outcome","clip_duration"
]

def write_csv(rows, csv_path: pathlib.Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_HEADER})

def write_json(obj, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--team", required=True)
    ap.add_argument("--playbook", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--min-play-length", type=float, default=3.0)
    ap.add_argument("--min-play-gap", type=float, default=1.5)
    ap.add_argument("--generate-report", action="store_true", default=True)
    ap.add_argument("--generate-clips", action="store_true", default=True)
    ap.add_argument("--highlights", action="store_true", default=True)
    ap.add_argument("--overlay", action="store_true", default=False)
    return ap

def main(argv=None):
    args = build_argparser().parse_args(argv)
    video_path = pathlib.Path(args.video).resolve()
    out_dir = pathlib.Path(args.out).resolve()
    game_dir = out_dir / "games" / f"{video_path.stem}__{hex(abs(hash(video_path)))[:12].replace('x','')}"
    game_dir.mkdir(parents=True, exist_ok=True)

    print(f"[playbook] source={pathlib.Path(args.playbook).resolve()}")
    with open(args.playbook) as f:
        playbook = json.load(f)
    print(f"[playbook] OK: loaded playbook from {args.playbook}")
    print("[config] min_play_length={} min_play_gap={} report={} clips={} highlights={} overlay={}"
          .format(args.min_play_length, args.min_play_gap, args.generate_report, args.generate_clips, args.highlights, args.overlay))

    # Step 1: segment
    segments = segment_video(str(video_path), min_play_length=args.min_play_length, min_gap=args.min_play_gap)
    print(f"[pipeline] segments detected: {len(segments)}")

    # Step 2: classify (returns list of per-play dicts with keys including candidates[])
    plays = classify_plays(segments)

    # Normalize rows for CSV, and build report with candidates
    csv_rows = []
    report = {"video": str(video_path), "plays": []}
    for p in plays:
        row = {
            "play_id": p.get("play_id"),
            "t0": p.get("t0"), "t1": p.get("t1"),
            "snap": p.get("snap"), "whistle": p.get("whistle"),
            "clip_path": p.get("clip_path", ""),
            "formation": p.get("formation", "Unknown"),
            "formation_confidence": round(float(p.get("formation_confidence", 0.0)), 2),
            "play_family": p.get("play_family", "Unknown"),
            "playcall_confidence": round(float(p.get("playcall_confidence", 0.0)), 2),
            "outcome": p.get("outcome", ""),
            "clip_duration": p.get("clip_duration", 0.0),
        }
        csv_rows.append(row)
        # Put candidates + all fields in report
        rp = dict(p)
        rp["candidates"] = p.get("candidates", [])  # ensure present
        report["plays"].append(rp)

    # Write outputs
    plays_csv = game_dir / "plays_index.csv"
    write_csv(csv_rows, plays_csv)
    if args.generate_report:
        report_json = game_dir / "report.json"
        write_json(report, report_json)

    # Optional clip generation
    if args.generate_clips:
        import subprocess, csv as _csv
        clips_dir = os.path.join(game_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        with open(plays_csv, "r", newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                play_id = row["play_id"]
                t0 = float(row["t0"])
                t1 = float(row["t1"])
                src = args.video
                out_dir = os.path.join(clips_dir, play_id)
                os.makedirs(out_dir, exist_ok=True)
                out_mp4 = os.path.join(out_dir, f"{play_id}.mp4")
                if not os.path.exists(out_mp4):
                    subprocess.run([
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-ss", f"{t0}", "-to", f"{t1}", "-i", src,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", out_mp4
                    ], check=True)
    print(f"[pipeline] run complete -> {game_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

