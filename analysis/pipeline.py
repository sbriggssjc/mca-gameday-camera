"""End-to-end orchestration for automated film analysis."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from . import (
    detect_track,
    play_recognizer,
    assignments,
    player_identity,
    grading,
    clipping,
    segmentation,
    formation_classifier,
    defense_grader,
    report_builder,
    io_utils,
)


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
    fps: int = 12,
    generate_report: bool = True,
    generate_clips: bool = True,
    generate_highlights: bool = True,
    min_play_gap: float = 7.0,
    min_play_length: float = 4.0,
    player_ids: str | None = None,
    id_overrides: str | None = None,
    team_color: str | None = None,
    grading_weights: str | None = None,
    clip_corrections: bool = False,
    clip_wins: bool = False,
    clip_highlights: bool = False,
) -> None:
    """Execute the toy analysis pipeline."""

    os.makedirs(out_dir, exist_ok=True)

    logger = logging.getLogger("pipeline")

    tracks = detect_track.run(video, team=team, fps=fps)
    detect_track.write_jsonl(tracks, os.path.join(out_dir, "tracking.jsonl"))

    identity_map = {t.player_id: t.player_id for t in tracks}
    if player_ids and os.path.exists(player_ids):
        signatures = player_identity.build_visual_signature_bank(player_ids)
        identity_map = player_identity.attach_identities_to_tracks(
            tracks, signatures, team_color or team, id_overrides
        )

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------
    dummy_frames = [None] * (fps * 10)
    segments = segmentation.segment_video(
        dummy_frames, fps, min_play_gap=min_play_gap, min_play_length=min_play_length, logger=logger
    )

    formations: List[str] = []
    playbook = assignments.load_playbook(playbook_path)
    for seg in segments:
        f = formation_classifier.classify_formation(playbook, [], fps)
        formations.append(f)

    # Build play-like structures for recogniser compatibility
    plays_dicts: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments, 1):
        plays_dicts.append(
            {
                "play_id": idx,
                "start_s": seg.start_ts,
                "end_s": seg.end_ts,
                "offense_color": team,
                "defense_color": "DARK",
                "hash_features": {"formation": formations[idx - 1]},
            }
        )
    _write_jsonl(plays_dicts, os.path.join(out_dir, "plays.jsonl"))

    preds = play_recognizer.recognize(
        plays_dicts, [pl.to_dict() for pl in playbook.offense_plays]
    )
    _write_jsonl(preds, os.path.join(out_dir, "play_predictions.jsonl"))

    player_grades = grading.grade(preds, tracks, identity_map, playbook, grading_weights)

    # Defensive grading output
    defense_grades = defense_grader.grade_plays(
        segments, formations, out_dir, grading_weights=grading_weights
    )

    # Metadata
    meta_path = Path(out_dir) / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    meta = {
        "team": team or meta.get("team") or "Metro Christian Academy",
        "opponent": meta.get("opponent", "UNKNOWN"),
        "video": video,
        "fps": fps,
        "play_count": len(segments),
    }
    io_utils.write_json(meta_path, meta)

    if generate_report:
        report_builder.build(
            out_dir=Path(out_dir),
            metadata_path=meta_path,
            formations=formations,
            segments=segments,
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Automated film analysis pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--team", default="WHITE")
    parser.add_argument("--playbook", default=None)
    parser.add_argument("--out", default="output")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--detect-model")
    parser.add_argument("--ocr", default="tesseract")
    parser.add_argument("--min-grade-good", type=float, default=2.5)
    parser.add_argument("--max-grade-needs", type=float, default=1.5)
    parser.add_argument("--generate-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generate-clips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generate-highlights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-play-gap", type=float, default=7.0)
    parser.add_argument("--min-play-length", type=float, default=4.0)
    parser.add_argument("--debug-vid", action="store_true")
    parser.add_argument("--player-ids")
    parser.add_argument("--id-overrides")
    parser.add_argument("--team-color")
    parser.add_argument("--grading-weights")
    parser.add_argument("--clip-corrections", action="store_true")
    parser.add_argument("--clip-wins", action="store_true")
    parser.add_argument("--clip-highlights", action="store_true")
    args = parser.parse_args(argv)

    run_pipeline(
        video=args.video,
        team=args.team,
        playbook_path=args.playbook,
        out_dir=args.out,
        fps=args.fps,
        generate_report=args.generate_report,
        generate_clips=args.generate_clips,
        generate_highlights=args.generate_highlights,
        min_play_gap=args.min_play_gap,
        min_play_length=args.min_play_length,
        player_ids=args.player_ids,
        id_overrides=args.id_overrides,
        team_color=args.team_color,
        grading_weights=args.grading_weights,
        clip_corrections=args.clip_corrections,
        clip_wins=args.clip_wins,
        clip_highlights=args.clip_highlights,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
