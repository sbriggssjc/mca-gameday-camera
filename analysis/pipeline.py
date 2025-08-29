from __future__ import annotations

import os, sys, json, pathlib, argparse, csv, subprocess

try:
    # When executed as module (recommended)
    from .segmentation import segment_video
    from .play_classifier import classify_plays
    from .playbook import load_playbook
except ImportError:  # pragma: no cover - fallback for script execution
    # Fallback when run as a script from repo root
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
    from analysis.segmentation import segment_video
    from analysis.play_classifier import classify_plays
    from analysis.playbook import load_playbook


def _ffmpeg(*args: str) -> subprocess.CompletedProcess:
    cmd = ["ffmpeg", "-y", *args]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in s)


def run_pipeline(
    video: str,
    team: str,
    playbook_path: str,
    out_dir: str,
    min_play_gap: float = 1.5,
    min_play_length: float = 3.0,
    max_play_length: float = 12.0,
    preroll: float = 0.75,
    postroll: float = 0.75,
    generate_report: bool = False,
    generate_clips: bool = False,
    debug_weak: bool = False,
) -> str:
    video = os.path.abspath(video)
    out_dir = os.path.abspath(out_dir)

    tag = pathlib.Path(video).stem
    short = hex(abs(hash((tag, playbook_path, min_play_gap, min_play_length))) & ((1 << 44) - 1))[2:]
    run_dir = os.path.join(out_dir, "games", f"{_safe_name(tag)}__{short}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "clips"), exist_ok=True)

    playbook = load_playbook(playbook_path)
    print(f"[playbook] source={playbook_path}")

    segments = segment_video(
        video,
        min_play_length=min_play_length,
        max_play_length=max_play_length,
        min_play_gap=min_play_gap,
        preroll=preroll,
        postroll=postroll,
    )
    print(
        f"[config] min_play_length={min_play_length} max_play_length={max_play_length} min_play_gap={min_play_gap} "
        f"report={generate_report} clips={generate_clips}"
    )
    print(f"[pipeline] segments detected: {len(segments)}")

    classifications = classify_plays(segments, playbook, team)

    rows: list[dict] = []
    for seg, det in zip(segments, classifications):
        pid = det.get("play_id") or f"PLAY_{len(rows)+1:03d}"
        rows.append(
            {
                "play_id": pid,
                "t0": float(seg["t0"]),
                "t1": float(seg["t1"]),
                "snap": float(seg.get("snap", seg["t0"])),
                "whistle": float(seg.get("whistle", seg["t1"])),
                "clip_path": "",
                "formation": det.get("formation", "Unknown"),
                "formation_confidence": float(det.get("formation_confidence", 0.0)),
                "play_family": det.get("play_family", "Unknown"),
                "playcall_confidence": float(det.get("playcall_confidence", 0.0)),
                # Observability fields
                "clf_top1": det.get("clf_top1", det.get("play_family", "")),
                "clf_top1_conf": float(det.get("clf_top1_conf", det.get("playcall_confidence", 0.0))),
                "clf_top3": ";".join(
                    f"{n}:{s:.2f}" for n, s in det.get("clf_top3", det.get("candidates", []))
                ),
                "clf_weak_flag": int(det.get("clf_weak_flag", 0)),
                "clf_family": det.get("clf_family", ""),
                "outcome": det.get("outcome") or "",
                "clip_duration": max(0.0, float(seg["t1"]) - float(seg["t0"])),
                "candidates": ";".join(
                    f"{n}:{s:.2f}" for n, s in det.get("candidates", [])
                ),
            }
        )

    csv_header = [
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
        "clf_top1",
        "clf_top1_conf",
        "clf_top3",
        "clf_weak_flag",
        "clf_family",
        "outcome",
        "clip_duration",
        "candidates",
    ]
    csv_path = os.path.join(run_dir, "plays_index.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if generate_clips:
        for r in rows:
            pid = r["play_id"]
            pdir = os.path.join(run_dir, "clips", pid)
            os.makedirs(pdir, exist_ok=True)
            t0, t1 = float(r["t0"]), float(r["t1"])
            mp4 = os.path.join(pdir, f"{pid}.mp4")
            dur = max(0.1, t1 - t0)
            proc = _ffmpeg(
                "-ss",
                f"{t0:.3f}",
                "-to",
                f"{t1:.3f}",
                "-i",
                video,
                "-c",
                "copy",
                mp4,
            )
            if proc.returncode != 0:
                _ffmpeg(
                    "-ss",
                    f"{t0:.3f}",
                    "-to",
                    f"{t1:.3f}",
                    "-i",
                    video,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-c:a",
                    "copy",
                    mp4,
                )
            r["clip_path"] = mp4

        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_header)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Optional debug frames for weak classifications
    if debug_weak:
        dbg_dir = os.path.join(run_dir, "debug", "weak")
        os.makedirs(dbg_dir, exist_ok=True)
        for idx, (seg, det) in enumerate(zip(segments, classifications), 1):
            if det.get("clf_weak_flag"):
                times = [
                    float(seg["t0"]),
                    float(seg["t0"] + seg["t1"]) / 2.0,
                    float(seg["t1"]),
                ]
                for j, t in enumerate(times):
                    _ffmpeg(
                        "-ss",
                        f"{t:.3f}",
                        "-i",
                        video,
                        "-frames:v",
                        "1",
                        os.path.join(dbg_dir, f"seg_{idx}_{j}.jpg"),
                    )

    if generate_report:
        report = {
            "video": video,
            "run_dir": run_dir,
            "n_segments": len(segments),
            "generated_clips": bool(generate_clips),
            "plays": rows,
        }
        with open(os.path.join(run_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)

    # QA guardrails
    if rows:
        long = [r for r in rows if r["clip_duration"] > max_play_length]
        unknown = [r for r in rows if not r["formation"] or not r["play_family"]]
        if len(long) / len(rows) > 0.25:
            print("⚠️ segmentation too coarse")
        if len(unknown) / len(rows) > 0.5:
            print("⚠️ formation/classifier weak")

    print(f"[pipeline] run complete -> {run_dir}")

    # Update "latest" symlinks for this video
    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "update_latest_symlinks.sh"
    try:
        subprocess.run(
            [
                "bash",
                str(script),
                tag,
                out_dir,
            ],
            check=False,
        )
    except Exception as e:  # pragma: no cover - best effort only
        print(f"[pipeline] symlink update failed: {e}")
    return run_dir


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--team", required=True)
    p.add_argument("--playbook", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-play-gap", type=float, default=1.5)
    p.add_argument("--min-play-length", type=float, default=3.0)
    p.add_argument("--max-play-length", type=float, default=12.0)
    p.add_argument("--preroll", type=float, default=0.75)
    p.add_argument("--postroll", type=float, default=0.75)
    p.add_argument("--generate-report", action="store_true")
    p.add_argument("--generate-clips", action="store_true")
    p.add_argument("--debug-weak", action="store_true")
    args = p.parse_args(argv)

    run_pipeline(
        video=args.video,
        team=args.team,
        playbook_path=args.playbook,
        out_dir=args.out,
        min_play_gap=args.min_play_gap,
        min_play_length=args.min_play_length,
        max_play_length=args.max_play_length,
        preroll=args.preroll,
        postroll=args.postroll,
        generate_report=args.generate_report,
        generate_clips=args.generate_clips,
        debug_weak=args.debug_weak,
    )


if __name__ == "__main__":
    main()

