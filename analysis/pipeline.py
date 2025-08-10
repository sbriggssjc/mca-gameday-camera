"""End-to-end orchestration for automated film analysis.

The real project aims to process full game footage and produce player
grades and highlight clips.  The implementation below is intentionally
minimal; it wires together lightweight placeholder modules so that the
command line interface and data-flow can be exercised in unit tests
without requiring heavy multimedia dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import yaml

from . import detect_track, team_role_assign, play_segment, play_recognizer, assignments, grader, highlights
from reports import generate_coach_report


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
) -> None:
    """Execute the toy analysis pipeline.

    Each stage writes small JSON artefacts to ``out_dir``.  The function is
    deliberately simple but mirrors the flow of the production system so unit
    tests can validate behaviour.
    """

    os.makedirs(out_dir, exist_ok=True)

    tracks = detect_track.run(video, team=team, fps=fps)
    detect_track.write_jsonl(tracks, os.path.join(out_dir, "tracking.jsonl"))

    plays = play_segment.segment([t.as_dict() for t in tracks])
    _write_jsonl([p.as_dict() for p in plays], os.path.join(out_dir, "plays.jsonl"))

    playbook = assignments.load_playbook(playbook_path)
    preds = play_recognizer.recognize([p.as_dict() for p in plays], playbook)
    _write_jsonl(preds, os.path.join(out_dir, "play_predictions.jsonl"))

    grades = []
    for pred in preds:
        assn = assignments.assignments_for_play(pred["predicted_play"], playbook)
        g = grader.grade_play(pred, assn)
        grades.append({"play_id": pred["play_id"], "grades": g})
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

    if generate_clips:
        # demonstrate directory structure creation
        highlights.ensure_output_dirs(out_dir, "10")
    if generate_report:
        generate_coach_report.generate(preds, grades, out_dir)


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
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
