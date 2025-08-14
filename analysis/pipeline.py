"""End-to-end orchestration for automated film analysis."""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

try:
    from overlays.debug_overlay import render_overlays_for_out_dir
except Exception as _e:  # pragma: no cover - optional dependency
    render_overlays_for_out_dir = None

try:
    from reporting.debug_summary import print_debug_summary
except Exception as _e:  # pragma: no cover - optional dependency
    print_debug_summary = None

from . import (
    tracker,
    assignments,
    player_identity,
    grading,
    clipping,
    formation_classifier,
    report_builder,
    play_matcher,
    grader,
    playbook_map,
    features,
    predict,
)
from .segmentation import Segment
from segment.play_segmenter import segment_video

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

DEFAULT_MIN_PLAY_LEN = 6.0
DEFAULT_MIN_PLAY_GAP = 1.5

PROFILE_DEFAULTS = {
    "game": {"min_play_length": 6.0, "min_play_gap": 1.5},
    "practice": {"min_play_length": 5.0, "min_play_gap": 1.0},
    "clinic": {"min_play_length": 4.0, "min_play_gap": 0.8},
}


@dataclass
class RunConfig:
    min_play_length: float
    min_play_gap: float
    strict: bool
    make_overlay: bool
    debug_summary: bool


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _normalize_video_if_needed(path: str) -> str:
    """Re-encode video to 720p landscape if portrait or not standard.

    Returns the path to the possibly re-encoded file. If the file does not
    exist or FFmpeg fails, the original path is returned unchanged.
    """

    if not os.path.exists(path):
        return path

    width = height = 0
    try:  # pragma: no cover - best effort only
        import cv2  # type: ignore

        cap = cv2.VideoCapture(path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
    except Exception:
        return path

    if width >= height and width == 1280 and height == 720:
        return path

    filters = []
    if height > width:
        filters.append("transpose=1")
    filters.append("scale=1280:720:flags=bicubic")
    vf = ",".join(filters)
    out_path = Path(path).with_name(Path(path).stem + "_720p_landscape.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        path,
        "-vf",
        vf,
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return str(out_path)
    except Exception:
        return path


def run_pipeline(
    video: str,
    team: str,
    playbook_path: str | None,
    out_dir: str,
    opponent: str | None = None,
    fps: int | None = None,
    generate_report: bool = True,
    generate_clips: bool = True,
    generate_highlights: bool = True,
    min_play_gap: float = DEFAULT_MIN_PLAY_GAP,
    min_play_length: float = DEFAULT_MIN_PLAY_LEN,
    player_ids: str | None = None,
    id_overrides: str | None = None,
    team_color: str | None = None,
    grading_weights: str | None = None,
    clip_corrections: bool = False,
    clip_wins: bool = False,
    clip_highlights: bool = False,
    detect_model: str | None = None,
    args: argparse.Namespace | None = None,
) -> None:
    """Execute the toy analysis pipeline."""

    os.makedirs(out_dir, exist_ok=True)

    # Normalize/rotate video when needed so detectors see a standard input
    video = _normalize_video_if_needed(video)

    logger = logging.getLogger("pipeline")

    # Open the video to gather metadata and detect FPS if not provided
    detected_fps = None
    frame_count = 0
    width = 0
    height = 0
    try:  # pragma: no cover - best effort only
        import cv2  # type: ignore

        cap = cv2.VideoCapture(video)
        detected_fps = cap.get(cv2.CAP_PROP_FPS) or None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
    except Exception:
        detected_fps = None

    # If portrait, swap W/H for downstream ROI math that assumes landscape
    if height > width:
        print(f"[video] Portrait detected ({width}x{height}); enabling portrait-safe ROI.")
        portrait = True
    else:
        portrait = False

    if fps in (None, 0):
        fps = detected_fps
    else:
        detected_fps = fps

    # --- FPS fallback ---
    if ((args is None) or getattr(args, "fps", None) in (None, 0, "")) and (
        detected_fps is None or detected_fps < 15
    ):
        fps = 30
        if logger:
            logger.warning(
                f"FPS fallback engaged: detected={detected_fps}, using fps={fps}"
            )

    fps = fps or 30
    duration_sec = (frame_count / fps) if (fps and frame_count) else 0.0

    meta = {
        "team": (args.team if args else team) or "",
        "opponent": (args.opponent if args else opponent) or "",
        "video_path": str(video),
        "video": str(video),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "video_length_sec": duration_sec,
    }
    meta_path = Path(out_dir) / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    segs = segment_video(
        video,
        fps,
        Path(out_dir),
        cfg={"min_play_length": min_play_length, "min_play_gap": min_play_gap},
        ctx={"video_length_sec": duration_sec},
    )
    print(f"[pipeline] Segments in memory: {len(segs)}")

    # Normalize segments with stable IDs
    plays: List[Dict[str, Any]] = []
    for i, s in enumerate(segs):
        p = dict(s)
        p.setdefault("segment_id", f"seg_{i:04d}")
        plays.append(p)
    _write_jsonl(plays, os.path.join(out_dir, "plays.jsonl"))

    # Update metadata with play count
    meta["play_count"] = len(plays)
    meta_path.write_text(json.dumps(meta, indent=2))

    settings_data: Dict[str, Any] = {}
    settings_fp = Path("config/settings.yaml")
    if settings_fp.exists() and yaml:
        try:
            loaded = yaml.safe_load(settings_fp.read_text()) or {}
            if isinstance(loaded, dict):
                settings_data = loaded
        except Exception:
            settings_data = {}

    tracks = tracker.track(video, plays, team=team, team_color=team_color)

    def _sid(d: Dict[str, Any]) -> Any:
        for k in ("segment_id", "id", "seg_id"):
            if k in d:
                return d[k]
        return None

    rows_by_sid = { _sid(r): r for r in (tracks or []) if _sid(r) is not None }
    safe_rows: List[Dict[str, Any]] = []
    for s in plays:
        sid = _sid(s)
        r = rows_by_sid.get(sid)
        if not r:
            r = {"segment_id": sid, "players": [], "meta": {"note": "empty_tracking"}}
        else:
            r["segment_id"] = r.get("segment_id") or r.get("seg_id") or sid
            r.pop("seg_id", None)
        safe_rows.append(r)

    with open(Path(out_dir) / "tracking.jsonl", "w") as f:
        for r in safe_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[tracking] wrote {len(safe_rows)} rows -> {Path(out_dir)/'tracking.jsonl'}")

    identity_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Segmentation-dependent structures
    # ------------------------------------------------------------------
    frames = [None] * int(fps * 10)
    segments = [Segment(float(p["start_s"]), float(p["end_s"])) for p in plays]

    playbook = assignments.load_playbook(playbook_path)

    # Formation and play matching for all segments
    formations = formation_classifier.classify_all(segments, frames, fps, playbook)
    play_matches = play_matcher.match_all(segments, frames, fps, playbook)

    # Tracking grouped by segment
    tracking_by_segment = {r["segment_id"]: r for r in safe_rows}

    # ------------------------------------------------------------------
    # Feature computation and play classification with UNKNOWN support
    # ------------------------------------------------------------------
    meta_dims = {"width": width, "height": height}
    feats = features.compute_all(safe_rows, meta=meta_dims, min_players=3)
    with open(Path(out_dir) / "features.jsonl", "w") as f:
        for r in feats:
            f.write(json.dumps(r) + "\n")
    print(f"[features] wrote {len(feats)} rows -> {Path(out_dir)/'features.jsonl'}")

    class _DummyModel:
        def predict_proba(self, X):  # type: ignore[override]
            return [[0.34, 0.33, 0.33]]

    dummy_model = _DummyModel()
    label_map = {0: "Rit Dive", 1: "Rit Sweep", 2: "Pass"}

    def model_predict(vec: List[float]) -> tuple[str, float]:
        probs = dummy_model.predict_proba([vec])[0]
        idx = max(range(len(probs)), key=lambda i: probs[i])
        return label_map[idx], probs[idx]

    pred_rows = predict.predict_all(feats, model_predict)
    play_lookup = {p["segment_id"]: p for p in plays}
    for r in pred_rows:
        r["play_id"] = play_lookup.get(r["segment_id"], {}).get("play_id")
    _write_jsonl(pred_rows, os.path.join(out_dir, "play_predictions.jsonl"))

    # Audit distribution and confidence
    from collections import Counter

    dist = Counter([r["predicted_play"] for r in pred_rows])
    print("[audit] predicted counts:", dict(dist))

    confs = [r["confidence"] for r in pred_rows if isinstance(r.get("confidence"), (int, float))]
    if confs:
        confs_sorted = sorted(confs)
        mid = confs_sorted[len(confs_sorted) // 2]
        print(
            f"[audit] confidence n={len(confs)} min={confs_sorted[0]:.2f} "
            f"p50={mid:.2f} max={confs_sorted[-1]:.2f}"
        )

    pred_by_segment = {r["segment_id"]: r for r in pred_rows}

    # ------------------------------------------------------------------
    # Defensive grading
    # ------------------------------------------------------------------
    playbook_data = {}
    if playbook_path and os.path.exists(playbook_path):
        try:
            playbook_data = json.loads(Path(playbook_path).read_text())
        except Exception:
            playbook_data = {}
    if "offense" not in playbook_data:
        playbook_data = {"offense": {"plays": playbook_data.get("plays", [])}}
    play_index = playbook_map.build_play_index(playbook_data)

    grade_rows: List[Dict[str, Any]] = []
    for p in plays:
        seg_id = p["segment_id"]
        pred = pred_by_segment.get(seg_id, {})
        tracking = tracking_by_segment.get(seg_id)
        g = grader.grade_defense(p, pred, tracking, play_index)
        g.update({"segment_id": seg_id, "play_id": p.get("play_id")})
        grade_rows.append(g)

    _write_jsonl(grade_rows, os.path.join(out_dir, "grades.jsonl"))

    # Integrity checks
    grade_by_segment = {g["segment_id"]: g for g in grade_rows}
    missing_pred = [p["segment_id"] for p in plays if p["segment_id"] not in pred_by_segment]
    missing_grade = [p["segment_id"] for p in plays if p["segment_id"] not in grade_by_segment]
    if missing_pred:
        print(
            f"[WARN] predictions missing for segments: {missing_pred[:5]}"
            + (f" ... (+{len(missing_pred)-5} more)" if len(missing_pred) > 5 else "")
        )
    if missing_grade:
        print(
            f"[WARN] grades missing for segments: {missing_grade[:5]}"
            + (f" ... (+{len(missing_grade)-5} more)" if len(missing_grade) > 5 else "")
        )

    player_grades = grading.grade(pred_rows, [], identity_map, playbook, grading_weights)

    if generate_report:
        report_builder.build(
            out_dir=Path(out_dir),
            metadata_path=meta_path,
            segments=segments,
            formations=formations,
            play_matches=play_matches,
            grades_path=Path(out_dir) / "grades.jsonl",
        )

    if args and getattr(args, "debug_vid", False):
        from . import debug_overlay

        seg_dicts = [
            {"start_frame": int(seg.start_ts * fps), "end_frame": int(seg.end_ts * fps)}
            for seg in segments
        ]
        debug_overlay.build_debug_video(
            video_path=Path(video),
            out_dir=Path(out_dir),
            segments=seg_dicts,
            fps=fps,
            formations=formations,
            play_matches=play_matches,
            grades_path=Path(out_dir) / "grades.jsonl",
        )

    if generate_report and not (clip_corrections or clip_wins or clip_highlights):
        clip_corrections = clip_wins = clip_highlights = True

    if generate_clips:
        clip_corrections = clip_corrections or True
        clip_wins = clip_wins or True

    if clip_corrections or clip_wins or clip_highlights or generate_highlights:
        clipping.export_clips(
            player_grades,
            out_dir,
            corrections=clip_corrections,
            wins=clip_wins,
            highlights=clip_highlights or generate_highlights,
        )

    if generate_highlights:
        from .highlights import build_highlight

        try:
            build_highlight(Path(out_dir) / "clips", Path(out_dir) / "highlights")
        except Exception as exc:  # pragma: no cover - best effort only
            if logger:
                logger.warning("Highlight build failed: %s", exc)

    if generate_report:
        from .report_emergency import build_emergency_report

        try:
            build_emergency_report(Path(out_dir))
        except Exception as exc:  # pragma: no cover - best effort only
            if logger:
                logger.warning("Emergency report build failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Automated film analysis pipeline (supports profiles, strict checks, overlays)"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--team", default="WHITE")
    parser.add_argument("--opponent", type=str, default=None)
    parser.add_argument("--playbook", default=None)
    parser.add_argument("--out", default="output")
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--detect-model")
    parser.add_argument("--ocr", default="tesseract")
    parser.add_argument("--min-grade-good", type=float, default=2.5)
    parser.add_argument("--max-grade-needs", type=float, default=1.5)
    parser.add_argument("--generate-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generate-clips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generate-highlights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-play-gap", type=float, default=0.0)
    parser.add_argument("--min-play-length", type=float, default=0.0)
    parser.add_argument(
        "--profile",
        choices=["game", "practice", "clinic"],
        default="game",
        help="Preset tuning for segmentation thresholds (affects min-play-length/gap). Default: game",
    )
    parser.add_argument(
        "--debug-vid",
        action="store_true",
        help="Render a debug video with overlays",
    )
    parser.add_argument("--player-ids")
    parser.add_argument("--id-overrides")
    parser.add_argument("--team-color")
    parser.add_argument("--grading-weights")
    parser.add_argument("--clip-corrections", action="store_true")
    parser.add_argument("--clip-wins", action="store_true")
    parser.add_argument("--clip-highlights", action="store_true")
    parser.add_argument(
        "--make-overlay",
        action="store_true",
        help="Generate debug overlay videos per play (and optionally a stitched full overlay)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if suspicious output (e.g., too few plays for clip length, zero matches, etc.)",
    )
    parser.add_argument(
        "--debug-summary",
        "--debug_summary",
        dest="debug_summary",
        action="store_true",
        help="Print counts of plays, formations, matches, and average grades at the end",
    )
    args = parser.parse_args(argv)

    # --- Build RunConfig with layered precedence ---
    prof = PROFILE_DEFAULTS.get(args.profile, PROFILE_DEFAULTS["game"])

    env_len = os.getenv("MCA_MIN_PLAY_LEN")
    env_gap = os.getenv("MCA_MIN_PLAY_GAP")

    min_play_length = (
        args.min_play_length
        if getattr(args, "min_play_length", None) not in (None, 0)
        else float(env_len)
        if env_len
        else float(prof["min_play_length"]) if prof else DEFAULT_MIN_PLAY_LEN
    )

    min_play_gap = (
        args.min_play_gap
        if getattr(args, "min_play_gap", None) not in (None, 0)
        else float(env_gap)
        if env_gap
        else float(prof["min_play_gap"]) if prof else DEFAULT_MIN_PLAY_GAP
    )

    run_cfg = RunConfig(
        min_play_length=min_play_length,
        min_play_gap=min_play_gap,
        strict=bool(args.strict),
        make_overlay=bool(args.make_overlay),
        debug_summary=bool(getattr(args, "debug_summary", False)),
    )

    print(
        f"[config] profile={args.profile} min_play_length={run_cfg.min_play_length:.2f}s "
        f"min_play_gap={run_cfg.min_play_gap:.2f}s strict={run_cfg.strict} "
        f"overlay={run_cfg.make_overlay} summary={run_cfg.debug_summary}"
    )

    run_pipeline(
        video=args.video,
        team=args.team,
        opponent=args.opponent,
        playbook_path=args.playbook,
        out_dir=args.out,
        fps=args.fps,
        generate_report=args.generate_report,
        generate_clips=args.generate_clips,
        generate_highlights=args.generate_highlights,
        min_play_gap=run_cfg.min_play_gap,
        min_play_length=run_cfg.min_play_length,
        player_ids=args.player_ids,
        id_overrides=args.id_overrides,
        team_color=args.team_color,
        grading_weights=args.grading_weights,
        clip_corrections=args.clip_corrections,
        clip_wins=args.clip_wins,
        clip_highlights=args.clip_highlights,
        detect_model=args.detect_model,
        args=args,
    )

    # ---- Strict checks & overlays & summary ----
    out_dir = Path(args.out) if args.out else Path("output")
    plays_fp = out_dir / "plays.jsonl"
    predictions_fp = out_dir / "play_predictions.jsonl"
    grades_fp = out_dir / "grades.jsonl"
    tracking_fp = out_dir / "tracking.jsonl"
    metadata_fp = out_dir / "metadata.json"

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
    if run_cfg.strict:
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
                render_overlays_for_out_dir(out_dir)
            except Exception as e:  # pragma: no cover - best effort
                print(f"[WARN] Overlay rendering failed: {e}")

    # Debug summary
    if run_cfg.debug_summary:
        if print_debug_summary is None:
            print(
                "[WARN] --debug-summary requested but reporting.debug_summary not importable; skipping summary."
            )

        else:
            try:
                print_debug_summary(
                    out_dir,
                    plays,
                    predictions,
                    grades,
                    profile=args.profile,
                    min_len=run_cfg.min_play_length,
                    min_gap=run_cfg.min_play_gap,
                )
            except Exception as e:  # pragma: no cover - best effort
                print(f"[WARN] Debug summary failed: {e}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
