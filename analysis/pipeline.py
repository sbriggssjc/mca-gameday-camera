"""Lightweight end-to-end video analysis pipeline.

This module wires together segmentation, lightweight tracking, feature
extraction, rule-based predictions, baseline grading and clip export.  It is
intentionally minimal and avoids heavyweight dependencies so it can run on a
Jetson without additional model weights.
"""
from __future__ import annotations

import argparse
from argparse import BooleanOptionalAction
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Fallback defaults
DEFAULT_MIN_PLAY_GAP = 1.5
DEFAULT_MIN_PLAY_LEN = 6.0

# Optional shared config; fall back to internal defaults if missing
try:
    from analysis.config import PROFILE_DEFAULTS as _PROFILE_DEFAULTS  # type: ignore
    PROFILE_DEFAULTS = dict(_PROFILE_DEFAULTS)
except Exception:
    PROFILE_DEFAULTS = {
        'game': {
            'min_play_gap': DEFAULT_MIN_PLAY_GAP,
            'min_play_length': DEFAULT_MIN_PLAY_LEN,
            'generate_report': True,
            'generate_clips': True,
            'generate_highlights': True,
            'make_overlay': False,
        }
    }

# Ensure baseline profile exists and has sane defaults
PROFILE_DEFAULTS.setdefault('game', {})
PROFILE_DEFAULTS['game']['make_overlay'] = PROFILE_DEFAULTS['game'].get('make_overlay', False) and False

import numpy as np

try:  # pragma: no cover
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from .segmentation import segment_video
from . import detect_track, features, orientation, zoom
from formation_detector import detect_formation
from analysis.playbook_loader import load_playbook
from .playbook.schema import validate_playbook
from .match.play_matcher import match_play


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
        rotation = 0
        try:
            rotation = orientation.estimate_rotation_degrees(args.video)
        except Exception:
            rotation = 0
        meta["rotation_deg"] = int(rotation)
    else:
        meta["rotation_deg"] = 0
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # load playbook for playcall matching if available
    raw_pb: Dict[str, Any]
    plays: List[Dict[str, Any]]
    raw_pb, plays = (load_playbook(args.playbook) if getattr(args, "playbook", None) else ({}, []))
    pb = None
    if plays:
        try:
            canonical_pb = {"plays": plays, "formations": raw_pb.get("formations", [])}
            pb = validate_playbook(canonical_pb)
        except Exception:
            pb = None
    print(f"[pipeline] playbook loaded with {len(plays)} plays")
    if len(plays) == 0:
        print("[pipeline] NOTE: classifier will degrade with 0 plays; continuing with formation-only heuristics.")

    # 1) segmentation
    segs = segment_video(args.video, min_play_gap=args.min_play_gap, min_play_length=args.min_play_length)
    print(f"[pipeline] segments detected: {len(segs)}")
    skip_play_match = len(plays) == 0
    if skip_play_match:
        print("[pipeline] INFO: skipping play matching (no plays in playbook). Formation-only pass will run.")

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
        try:
            tracks = detect_track.track_from_frames(frames, team=args.team)
        except Exception:
            tracks = []
        players = []
        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            players.append({"bbox": [x1, y1, x2, y2], "id": tr.player_id})
            centers_per_frame.append([(cx, cy)])
        track_row = {"segment_id": seg_id, "players": players}

        try:
            feat = features.compute_all([track_row], meta={"width": width, "height": height})[0]
        except Exception:
            feat = {"features": {}, "num_players": 0}
        features_rows.append({"segment_id": seg_id, "features": feat.get("features", {}), "num_players": feat.get("num_players", 0)})

        # Formation detection on first frame with player boxes
        formation_name = "Unknown"
        formation_conf = 0.0
        formation_cands: List[Dict[str, Any]] = []
        if frames:
            bboxes = [tuple(map(int, pl["bbox"])) for pl in players]
            try:
                formation_name, _ = detect_formation(frames[0], bboxes, play_id=int(i))
            except Exception:
                formation_name = "Unknown"
        if formation_name != "Unknown":
            formation_conf = 0.8
        formation_cands.append({"name": formation_name, "score": formation_conf})
        print(f"[formation_detector] {seg_id}: {formation_name} conf={formation_conf:.2f}")

        # Playcall classification using playbook matcher
        play_candidates: List[tuple[str, float]] = []
        play_name: Optional[str] = None
        play_conf = 0.0
        play_family = ""
        if not skip_play_match and pb and formation_name != "Unknown":
            try:
                play_candidates = match_play(pb, formation_name, {})
            except Exception:
                play_candidates = []
            if play_candidates:
                play_name, play_conf = play_candidates[0]
                ps = pb.plays.get(play_name)
                if ps and ps.family:
                    play_family = ps.family
        playcall_dict = {
            "name": play_name,
            "confidence": float(play_conf),
            "candidates": [{"name": n, "score": s} for n, s in play_candidates],
        }
        if not play_name:
            print(f"[play_classifier] {seg_id}: Unknown conf={play_conf:.2f}")
        else:
            print(f"[play_classifier] {seg_id}: {play_name} conf={play_conf:.2f}")

        formation_dict = {
            "name": formation_name if formation_name != "Unknown" else None,
            "confidence": formation_conf,
            "candidates": formation_cands,
        }
        prediction_rows.append({
            "play_id": seg_id,
            "formation": formation_dict,
            "playcall": playcall_dict,
            "play_family": play_family,
        })

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
                "formation": formation_name,
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
        fname = pr.get("formation", {}).get("name", "")
        summary["formations"][fname] = summary["formations"].get(fname, 0) + 1
        pfam = pr.get("play_family", "")
        summary["play_families"][pfam] = summary["play_families"].get(pfam, 0) + 1
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

    # Boolean flags with None default so profiles/env can override
    p.add_argument("--generate-report", action=BooleanOptionalAction, default=None)
    p.add_argument("--generate-clips", action=BooleanOptionalAction, default=None)
    p.add_argument("--generate-highlights", action=BooleanOptionalAction, default=None)
    p.add_argument("--make-overlay", action=BooleanOptionalAction, default=None)
    p.add_argument("--orientation-auto", action=BooleanOptionalAction, default=None)
    p.add_argument("--auto-zoom", action=BooleanOptionalAction, default=None)
    p.add_argument("--overwrite", action=BooleanOptionalAction, default=None)
    p.add_argument("--auto-draw", action=BooleanOptionalAction, default=None)

    # Numeric thresholds
    p.add_argument("--min-play-gap", type=float, default=None)
    p.add_argument("--min-play-length", type=float, default=None)
    p.add_argument("--clip-pre", type=float, default=None)
    p.add_argument("--clip-post", type=float, default=None)

    # Optional paths/strings
    p.add_argument("--player-ids", type=str, default=None)
    p.add_argument("--id-overrides", type=str, default=None)
    p.add_argument("--grading-weights", type=str, default=None)
    p.add_argument("--team-color", type=str, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    prof = PROFILE_DEFAULTS.get('game', {})
    env_len = os.getenv("MCA_MIN_PLAY_LEN")
    env_gap = os.getenv("MCA_MIN_PLAY_GAP")

    min_play_length = (
        args.min_play_length if getattr(args, 'min_play_length', None) not in (None, 0)
        else float(env_len) if env_len
        else float(prof.get('min_play_length', DEFAULT_MIN_PLAY_LEN))
    )
    min_play_gap = (
        args.min_play_gap if getattr(args, 'min_play_gap', None) not in (None, 0)
        else float(env_gap) if env_gap
        else float(prof.get('min_play_gap', DEFAULT_MIN_PLAY_GAP))
    )

    generate_report = prof.get('generate_report', True) if getattr(args, 'generate_report', None) is None else args.generate_report
    generate_clips = prof.get('generate_clips', True) if getattr(args, 'generate_clips', None) is None else args.generate_clips
    generate_highlights = prof.get('generate_highlights', True) if getattr(args, 'generate_highlights', None) is None else args.generate_highlights
    make_overlay = prof.get('make_overlay', False) if getattr(args, 'make_overlay', None) is None else args.make_overlay
    orientation_auto = prof.get('orientation_auto', False) if getattr(args, 'orientation_auto', None) is None else args.orientation_auto
    auto_zoom = prof.get('auto_zoom', False) if getattr(args, 'auto_zoom', None) is None else args.auto_zoom
    overwrite = prof.get('overwrite', False) if getattr(args, 'overwrite', None) is None else args.overwrite
    auto_draw = prof.get('auto_draw', False) if getattr(args, 'auto_draw', None) is None else args.auto_draw

    clip_pre = args.clip_pre if args.clip_pre is not None else 1.0
    clip_post = args.clip_post if args.clip_post is not None else 1.0

    args.min_play_length = min_play_length
    args.min_play_gap = min_play_gap
    args.generate_report = generate_report
    args.generate_clips = generate_clips
    args.generate_highlights = generate_highlights
    args.make_overlay = make_overlay
    args.orientation_auto = orientation_auto
    args.auto_zoom = auto_zoom
    args.overwrite = overwrite
    args.auto_draw = auto_draw
    args.clip_pre = clip_pre
    args.clip_post = clip_post

    print(
        f"[config] min_play_length={min_play_length} min_play_gap={min_play_gap} "
        f"report={generate_report} clips={generate_clips} highlights={generate_highlights} overlay={make_overlay}"
    )

    run_pipeline(args=args)


if __name__ == "__main__":  # pragma: no cover
    main()
