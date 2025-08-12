"""End-to-end orchestration for automated film analysis."""
from __future__ import annotations

import argparse
import json
import logging
import os
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
    detect_track,
    play_recognizer,
    assignments,
    player_identity,
    grading,
    clipping,
    formation_classifier,
    defense_grader,
    report_builder,
    play_matcher,
)
from .segmentation import Segment
from segment.play_segmenter import segment_video


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
    args: argparse.Namespace | None = None,
) -> None:
    """Execute the toy analysis pipeline."""

    os.makedirs(out_dir, exist_ok=True)

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

    # Update metadata with play count
    meta["play_count"] = len(segs)
    meta_path.write_text(json.dumps(meta, indent=2))

    tracks = detect_track.run(video, team=team, fps=fps)
    detect_track.write_jsonl(tracks, os.path.join(out_dir, "tracking.jsonl"))

    identity_map = {t.player_id: t.player_id for t in tracks}
    if player_ids and os.path.exists(player_ids):
        signatures = player_identity.build_visual_signature_bank(player_ids)
        identity_map = player_identity.attach_identities_to_tracks(
            tracks, signatures, team_color or team, id_overrides
        )

    # ------------------------------------------------------------------
    # Segmentation-dependent structures
    # ------------------------------------------------------------------
    frames = [None] * int(fps * 10)
    segments = [Segment(float(s["start_s"]), float(s["end_s"])) for s in segs]

    playbook = assignments.load_playbook(playbook_path)

    # Formation and play matching for all segments
    formations = formation_classifier.classify_all(segments, frames, fps, playbook)
    play_matches = play_matcher.match_all(segments, frames, fps, playbook)

    # Build play-like structures for recogniser compatibility
    plays_dicts: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segs, 1):
        pd = dict(seg)
        pd.update(
            {
                "offense_color": team,
                "defense_color": "DARK",
                "hash_features": {"formation": formations[idx - 1]},
            }
        )
        plays_dicts.append(pd)

    preds = play_recognizer.recognize(
        plays_dicts, [pl.to_dict() for pl in playbook.offense_plays]
    )
    _write_jsonl(preds, os.path.join(out_dir, "play_predictions.jsonl"))

    player_grades = grading.grade(preds, tracks, identity_map, playbook, grading_weights)

    # Defensive grading output
    defense_grader.grade_plays(
        segments,
        frames,
        fps,
        out_dir,
        formations=formations,
        grading_weights=grading_weights,
    )

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
