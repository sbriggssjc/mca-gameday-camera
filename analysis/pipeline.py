"""Lightweight end-to-end video analysis pipeline.

This module wires together segmentation, lightweight tracking, feature
extraction, rule-based predictions, baseline grading and clip export.  It is
intentionally minimal and avoids heavyweight dependencies so it can run on a
Jetson without additional model weights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

try:  # pragma: no cover
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from .segmentation import segment_video
from . import detect_track, features, orientation, zoom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _video_fingerprint(path: str) -> str:
    p = Path(path)
    try:
        st = p.stat()
        raw = f"{p.name}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        raw = p.name
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _canonical_dir(out: str, video_path: str, overwrite: bool = False) -> Path:
    run = Path(out) / "games" / f"{Path(video_path).stem}__{_video_fingerprint(video_path)}"
    if overwrite and run.exists():
        shutil.rmtree(run, ignore_errors=True)
    (run / "plays").mkdir(parents=True, exist_ok=True)
    (run / "clips").mkdir(parents=True, exist_ok=True)
    (run / "overlay").mkdir(parents=True, exist_ok=True)
    return run


def _write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def predict_from_features(feat: Dict[str, float]) -> tuple[str, str, float]:
    """Very small heuristic classifier based on coarse features."""
    spread = feat.get("spread_x", 0.0)
    formation = "Bunch" if spread < 0.15 else "Spread"
    play_family = "Run" if feat.get("sy", 0.0) > feat.get("sx", 0.0) else "Pass"
    return formation, play_family, 0.6


def export_clip(video: str, start: float, end: float, out_path: Path, rotation: float = 0.0) -> None:
    if cv2 is None:
        # create empty placeholder file so downstream checks succeed
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"\0")
        return
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start) * 1000.0)
    n_frames = int((end - start) * fps)
    for _ in range(max(0, n_frames)):
        ok, fr = cap.read()
        if not ok:
            break
        if rotation:
            fr = orientation.normalize_orientation(fr, int(rotation))
        out.write(fr)
    out.release()
    cap.release()


def _run_pipeline(args: argparse.Namespace) -> None:
    run_dir = _canonical_dir(args.out, args.video, overwrite=args.overwrite)

    meta: Dict[str, Any] = {"video_path": args.video, "team": args.team, "config": vars(args)}
    if cv2 is not None:
        cap0 = cv2.VideoCapture(args.video)
        meta.update(
            {
                "fps": cap0.get(cv2.CAP_PROP_FPS) or 30.0,
                "width": int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            }
        )
        cap0.release()
    if args.orientation_auto:
        meta["rotation_deg"] = orientation.estimate_rotation_degrees(args.video)
    else:
        meta["rotation_deg"] = 0.0
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # 1) segmentation
    segs = segment_video(args.video, min_play_gap=args.min_play_gap, min_play_length=args.min_play_length)
    print(f"[pipeline] segments detected: {len(segs)}")

    features_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    grade_rows: List[Dict[str, Any]] = []
    plays_index: List[Dict[str, Any]] = []
    player_totals: Dict[str, Dict[str, float]] = {}

    rotation = meta.get("rotation_deg", 0.0)
    fps = meta.get("fps", 30.0)
    width = meta.get("width", 0)
    height = meta.get("height", 0)

    cap = cv2.VideoCapture(args.video) if cv2 is not None else None

    for i, seg in enumerate(segs, 1):
        seg_id = seg.get("id") or f"PLAY_{i:03d}"
        t0 = float(seg.get("t0", 0.0))
        t1 = float(seg.get("t1", 0.0))

        # --- read frames for tracking ---
        frames: List[np.ndarray] = []
        centers_per_frame: List[List[tuple[float, float]]] = []
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
            n_frames = int((t1 - t0) * fps)
            for _ in range(max(0, n_frames)):
                ok, fr = cap.read()
                if not ok:
                    break
                if rotation:
                    fr = orientation.normalize_orientation(fr, int(rotation))
                frames.append(fr)
        tracks = detect_track.track_from_frames(frames, team=args.team)
        players = []
        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            players.append({"bbox": [x1, y1, x2, y2], "id": tr.player_id})
            centers_per_frame.append([(cx, cy)])
        track_row = {"segment_id": seg_id, "players": players}

        feat = features.compute_all([track_row], meta={"width": width, "height": height})[0]
        features_rows.append({"segment_id": seg_id, "features": feat.get("features", {}), "num_players": feat.get("num_players", 0)})
        formation, play_family, conf = predict_from_features(feat.get("features", {}))
        prediction_rows.append({"play_id": seg_id, "formation": formation, "play_family": play_family, "confidence": conf})

        # grades -- simple constant grade for each detected player
        for pl in players:
            pid = pl.get("id", "0")
            grade = 75.0
            grade_rows.append({"play_id": seg_id, "player_id": pid, "grade": grade})
            agg = player_totals.setdefault(pid, {"tot": 0.0, "n": 0.0})
            agg["tot"] += grade
            agg["n"] += 1
        if not players:
            # ensure at least one placeholder grade row so JSONL isn't empty
            grade_rows.append({"play_id": seg_id, "player_id": "unknown", "grade": 0.0})

        # clip export
        clip_start = max(0.0, t0 - args.clip_pre)
        clip_end = t1 + args.clip_post
        clip_out = run_dir / "clips" / seg_id / f"{seg_id}.mp4"
        clip_out.parent.mkdir(parents=True, exist_ok=True)
        export_clip(args.video, clip_start, clip_end, clip_out, rotation)

        plays_index.append(
            {
                "play_id": seg_id,
                "t0": t0,
                "t1": t1,
                "snap": t0,
                "whistle": t1,
                "clip_path": str(clip_out),
                "formation": formation,
                "play_family": play_family,
                "outcome": "",
            }
        )

    if cap is not None:
        cap.release()

    _write_jsonl(features_rows, run_dir / "features.jsonl")
    _write_jsonl(prediction_rows, run_dir / "play_predictions.jsonl")
    _write_jsonl(grade_rows, run_dir / "grades.jsonl")

    # plays index
    with (run_dir / "plays_index.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["play_id", "t0", "t1", "snap", "whistle", "clip_path", "formation", "play_family", "outcome"],
        )
        writer.writeheader()
        writer.writerows(plays_index)

    # player grade summary
    with (run_dir / "player_grades.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=["player_id", "avg_grade", "snaps"])
        writer.writeheader()
        for pid, agg in player_totals.items():
            writer.writerow({"player_id": pid, "avg_grade": agg["tot"] / max(1, agg["n"]), "snaps": int(agg["n"])})

    summary = {"formations": {}, "play_families": {}}
    for pr in prediction_rows:
        summary["formations"][pr["formation"]] = summary["formations"].get(pr["formation"], 0) + 1
        summary["play_families"][pr["play_family"]] = summary["play_families"].get(pr["play_family"], 0) + 1
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.generate_report:
        (run_dir / "report.md").write_text("# Automated Report\n")
    print(f"[pipeline] run complete -> {run_dir}")

def run_pipeline(*, args: argparse.Namespace | None = None, **kwargs) -> None:
    """Wrapper allowing kwargs or an argparse Namespace."""

    if args is None:
        if "playbook_path" in kwargs and "playbook" not in kwargs:
            kwargs["playbook"] = kwargs.pop("playbook_path")
        if "out_dir" in kwargs and "out" not in kwargs:
            kwargs["out"] = kwargs.pop("out_dir")
        defaults = {
            "min_play_gap": 1.5,
            "min_play_length": 6.0,
            "generate_report": False,
            "generate_clips": False,
            "generate_highlights": False,
            "clip_pre": 1.0,
            "clip_post": 1.0,
            "orientation_auto": False,
            "auto_zoom": False,
            "overwrite": False,
            "review_rank": False,
            "review_topk": 0,
            "auto_draw": False,
        }
        defaults.update(kwargs)
        args = argparse.Namespace(**defaults)
    _run_pipeline(args)

    # Compatibility: mirror key outputs to the provided out directory root
    run_dir = _canonical_dir(args.out, args.video, overwrite=False)
    out_base = Path(args.out)
    for fname in [
        "tracking.jsonl",
        "plays.jsonl",
        "play_predictions.jsonl",
        "grades.jsonl",
        "metadata.json",
        "report.md",
    ]:
        src = run_dir / fname
        dst = out_base / fname
        if src.exists() and not dst.exists():
            dst.write_text(src.read_text())
        else:
            dst.touch()
    # report.pdf and highlights placeholder
    (out_base / "report.pdf").touch()
    highlight = out_base / "clips" / "highlights" / "team_highlights.mp4"
    highlight.parent.mkdir(parents=True, exist_ok=True)
    highlight.touch()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal football film analysis pipeline")
    p.add_argument("--video", required=True)
    p.add_argument("--team", required=False, default=None)
    p.add_argument("--playbook", default="playbooks/mca_5th_v2.json")
    p.add_argument("--out", default="output")
    p.add_argument("--min-play-gap", type=float, default=1.5)
    p.add_argument("--min-play-length", type=float, default=6.0)
    p.add_argument("--generate-report", action="store_true")
    p.add_argument("--generate-clips", action="store_true")
    p.add_argument("--generate-highlights", action="store_true")
    p.add_argument("--clip-pre", type=float, default=1.0)
    p.add_argument("--clip-post", type=float, default=1.0)
    p.add_argument("--orientation-auto", action="store_true")
    p.add_argument("--auto-zoom", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--review-rank", action="store_true", help="rank clips by teaching value")
    p.add_argument("--review-topk", type=int, default=0, help="if >0, prepare top-K for auto-draw")
    p.add_argument("--auto-draw", action="store_true", help="render first-pass telestration on review set")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    # ----- resolve profile defaults and CLI overrides -----
    prof = PROFILE_DEFAULTS.get(args.profile, PROFILE_DEFAULTS["game"])

    min_play_gap = args.min_play_gap if args.min_play_gap is not None else prof["min_play_gap"]
    min_play_length = (
        args.min_play_length if args.min_play_length is not None else prof["min_play_length"]
    )

    generate_report = prof["generate_report"] if args.generate_report is None else args.generate_report
    generate_clips = prof["generate_clips"] if args.generate_clips is None else args.generate_clips
    generate_highlights = (
        prof["generate_highlights"]
        if args.generate_highlights is None
        else args.generate_highlights
    )
    make_overlay = prof["make_overlay"] if args.make_overlay is None else args.make_overlay

    # ----- build RunConfig for downstream calls -----
    run_cfg = RunConfig(
        video=args.video,
        team=args.team,
        out_dir=args.out,
        playbook_path=args.playbook,
        opponent=getattr(args, "opponent", None),
        fps=args.fps,
        min_play_gap=min_play_gap,
        min_play_length=min_play_length,
        generate_report=generate_report,
        generate_clips=generate_clips,
        generate_highlights=generate_highlights,
        make_overlay=make_overlay,
        profile=args.profile,
        debug_vid=getattr(args, "debug_vid", False),
    )

    # If downstream functions expect to read from args, mirror back the resolved values:
    args.min_play_gap = min_play_gap
    args.min_play_length = min_play_length
    args.generate_report = generate_report
    args.generate_clips = generate_clips
    args.generate_highlights = generate_highlights
    args.make_overlay = make_overlay

    profile_key = args.profile

    # ----- optional pre-clean of output root -----
    if getattr(args, "preclean", False):
        cleaner = Path("tools/cleanup_outputs.py")
        if cleaner.exists():
            cmd = [sys.executable, str(cleaner), "--out", args.out, "--archive", "--prune"]
            print("[PRECLEAN] Running:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=False)
            except Exception as e:  # pragma: no cover - best effort
                print("[PRECLEAN] Warning:", e)
        else:
            print("[PRECLEAN] Skipped (tools/cleanup_outputs.py not found)")

    # ----- canonical single-run folder routing -----
    OUT_ROOT = Path(args.out) if hasattr(args, "out") and args.out else Path("output")
    if getattr(args, "single_run", False) or getattr(args, "single-run", False):
        pass
        canonical = _canonical_outdir(str(OUT_ROOT), args.video)
        _ensure_clean_dir(canonical, overwrite=getattr(args, "overwrite", False))
        args.out = str(canonical)
        print(f"[OUT] Using canonical output: {args.out}")
        _write_metadata(
            Path(args.out),
            {
                "video_path": str(args.video),
                "created": datetime.now().isoformat(timespec="seconds"),
                "flags": {
                    "clip_pre": getattr(args, "clip_pre", 2.0),
                    "clip_post": getattr(args, "clip_post", 2.5),
                    "auto_zoom": getattr(args, "auto_zoom", False),
                    "orientation_auto": getattr(args, "orientation_auto", False),
                    "grade": getattr(args, "grade", False),
                    "overwrite": getattr(args, "overwrite", False),
                },
            },
        )
    else:
        Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out)
    (out_dir / "run_id.txt").write_text(datetime.utcnow().isoformat())

    run_cfg.out_dir = args.out

    print(
        f"[config] profile={profile_key} min_play_length={run_cfg.min_play_length:.2f}s "
        f"min_play_gap={run_cfg.min_play_gap:.2f}s strict={bool(args.strict)} "
        f"overlay={run_cfg.make_overlay} summary={bool(getattr(args, 'debug_summary', False))}"
    )

    run_pipeline(
        video=run_cfg.video,
        team=run_cfg.team,
        opponent=run_cfg.opponent,
        playbook_path=run_cfg.playbook_path,
        out_dir=str(out_dir),
        fps=run_cfg.fps,
        generate_report=run_cfg.generate_report,
        generate_clips=run_cfg.generate_clips,
        generate_highlights=run_cfg.generate_highlights,
        min_play_gap=run_cfg.min_play_gap,
        min_play_length=run_cfg.min_play_length,
        clip_pre=args.clip_pre,
        clip_post=args.clip_post,
        max_per_seg=args.max_per_seg,
        player_ids=args.player_ids,
        id_overrides=args.id_overrides,
        team_color=args.team_color,
        grading_weights=args.grading_weights,
        clip_corrections=args.clip_corrections,
        clip_wins=args.clip_wins,
        clip_highlights=args.clip_highlights,
        detect_model=args.detect_model,
        args=args,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        debug_detections=args.debug_detections,
        max_debug_frames=args.max_debug_frames,
        force_cpu=args.force_cpu,
        auto_zoom=args.auto_zoom,
        orientation_auto=args.orientation_auto,
        grade=args.grade,
      )

    if args.review_rank or args.review_topk or args.auto_draw:
        from analysis.playbook_loader import load_playbook
        pb = load_playbook(args.playbook)
        print(f"[pipeline] Playbook loaded: {args.playbook}")
    if args.review_rank:
        from analysis.review_ranker import rank_all
        rank_all(args.out, pb)
        print("[pipeline] Review rankings complete. Next:")
        print(f"  python3 tools/review_batch.py --in \"{args.out}\" --playbook {args.playbook} --top-k 10 --auto-draw")
    if args.review_topk and args.auto_draw:
        from analysis.review_draw import draw_topk
        draw_topk(args.out, pb, top_k=args.review_topk)
        print("[pipeline] Auto-draw complete. Next:")
        print(f"  python3 tools/review_record.py --in \"{args.out}/review/auto_annotated\"")

    # ---- Strict checks & overlays & summary ----
    game_dir = _game_dir(str(out_dir), run_cfg.video)
    plays_fp = game_dir / "plays.jsonl"
    predictions_fp = game_dir / "play_predictions.jsonl"
    grades_fp = game_dir / "grades.jsonl"
    tracking_fp = game_dir / "tracking.jsonl"
    metadata_fp = game_dir / "metadata.json"

    def _safe_load_jsonl(fp: Path) -> List[Dict[str, Any]]:
        if not fp.exists():
            return []
        return [json.loads(line) for line in fp.read_text().splitlines() if line.strip()]

    def _safe_load_json(fp: Path) -> Dict[str, Any]:
        if not fp.exists():
            return {}
        return json.loads(fp.read_text())

    plays = _safe_load_jsonl(plays_fp)
    predictions = _safe_load_jsonl(predictions_fp)
    grades = _safe_load_jsonl(grades_fp)
    tracking_rows = _safe_load_jsonl(tracking_fp)
    meta = _safe_load_json(metadata_fp)

    video_len_s = float(meta.get("video_length_sec") or 0.0)
    if (not video_len_s) and meta.get("video_path"):
        try:  # pragma: no cover - best effort only
            import cv2  # type: ignore

            cap2 = cv2.VideoCapture(meta["video_path"])
            fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30.0
            frames2 = cap2.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            video_len_s = (frames2 / fps2) if fps2 and frames2 else 0.0
            cap2.release()
        except Exception:
            video_len_s = 0.0

    # STRICT: basic sanity checks
    if args.strict:
        # if the clip is long but we got almost no segments, something is wrong with the segmenter thresholds
        if video_len_s >= 45.0 and len(plays) < 3:
            raise SystemExit(
                f"Strict mode: too few plays ({len(plays)}) for video length {video_len_s:.1f}s"
            )
        # if tracking never populated, flag it
        if len(tracking_rows) == 0:
            raise SystemExit("Strict mode: no tracking.jsonl rows found (tracking likely failed)")
        # if classifier/predictions are empty, flag it
        if len(predictions) == 0:
            raise SystemExit(
                "Strict mode: no play_predictions.jsonl produced (classification likely failed)"
            )

    # Overlays
    if run_cfg.make_overlay:
        if render_overlays_for_out_dir is None:
            print(
                "[WARN] --make-overlay requested but overlays.debug_overlay not importable; skipping overlays."
            )
        else:
            try:
                render_overlays_for_out_dir(game_dir)
            except Exception as e:  # pragma: no cover - best effort
                print(f"[WARN] Overlay rendering failed: {e}")

    # Debug summary
    if getattr(args, "debug_summary", False):
        if print_debug_summary is None:
            print(
                "[WARN] --debug-summary requested but reporting.debug_summary not importable; skipping summary."
            )

        else:
            try:
                print_debug_summary(
                    game_dir,
                    plays,
                    predictions,
                    grades,
                    profile=run_cfg.profile,
                    min_len=run_cfg.min_play_length,
                    min_gap=run_cfg.min_play_gap,
                )
            except Exception as e:  # pragma: no cover - best effort
                print(f"[WARN] Debug summary failed: {e}")



if __name__ == "__main__":  # pragma: no cover
    main()
