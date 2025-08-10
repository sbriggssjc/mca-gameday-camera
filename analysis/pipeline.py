"""End-to-end orchestration for automated film analysis."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from . import (
    detect_track,
    play_segment,
    play_recognizer,
    assignments,
    player_identity,
    grading,
    report,
    clipping,
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
    generate_report: bool = False,
    generate_clips: bool = False,
    generate_highlights: bool = False,
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

    tracks = detect_track.run(video, team=team, fps=fps)
    detect_track.write_jsonl(tracks, os.path.join(out_dir, "tracking.jsonl"))

    identity_map = {t.player_id: t.player_id for t in tracks}
    if player_ids and os.path.exists(player_ids):
        signatures = player_identity.build_visual_signature_bank(player_ids)
        identity_map = player_identity.attach_identities_to_tracks(
            tracks, signatures, team_color or team, id_overrides
        )

    plays = play_segment.segment([t.as_dict() for t in tracks])
    _write_jsonl([p.as_dict() for p in plays], os.path.join(out_dir, "plays.jsonl"))

    playbook = assignments.load_playbook(playbook_path)
    preds = play_recognizer.recognize(
        [p.as_dict() for p in plays], playbook["offense"]["plays"]
    )
    _write_jsonl(preds, os.path.join(out_dir, "play_predictions.jsonl"))

    grades = grading.grade(preds, tracks, identity_map, playbook, grading_weights)
    _write_jsonl(grades, os.path.join(out_dir, "grades.jsonl"))

    meta = {
        "game_id": "TEST",
        "video_path": video,
        "date": "1970-01-01",
        "team_us": team,
        "opponent": "UNKNOWN",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf8") as f:
        json.dump(meta, f)

    if generate_report:
        report.generate(grades, out_dir)

    if generate_report and not (clip_corrections or clip_wins or clip_highlights):
        clip_corrections = clip_wins = clip_highlights = True

    if generate_clips:
        clip_corrections = clip_corrections or True
        clip_wins = clip_wins or True

    if clip_corrections or clip_wins or clip_highlights or generate_highlights:
        clipping.export_clips(
            grades,
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
    parser.add_argument("--generate-report", action="store_true")
    parser.add_argument("--generate-clips", action="store_true")
    parser.add_argument("--generate-highlights", action="store_true")
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
