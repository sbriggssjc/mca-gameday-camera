"""Command line interface for film analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tools.json_io import load_json_safe

from .analyze import analyze_game
from .export import export_coach_summary, export_highlights, export_player_clips


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mca_film")
    sub = parser.add_subparsers(dest="cmd", required=True)

    analyze_p = sub.add_parser("analyze", help="run analysis on a video")
    analyze_p.add_argument("--video", required=True)
    analyze_p.add_argument("--side", choices=["offense", "defense"], default="offense")
    analyze_p.add_argument("--roster", default="config/roster.json")
    analyze_p.add_argument("--settings", default="config/settings.yaml")
    analyze_p.add_argument("--calibrate", action="store_true")
    analyze_p.add_argument("--min-confidence", type=float, default=0.72)
    analyze_p.add_argument("--export-clips", action="store_true")
    analyze_p.add_argument("--export-summary", action="store_true")
    analyze_p.add_argument("--export-highlights", action="store_true")

    export_p = sub.add_parser("export", help="export reports or clips")
    export_p.add_argument("--report", choices=["coaches"], nargs="?")
    export_p.add_argument("--players", nargs="*", default=[])
    export_p.add_argument("--highlights", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "analyze":
        import yaml

        roster = load_json_safe(Path(args.roster), default={})
        settings = yaml.safe_load(Path(args.settings).read_text())
        analyses = analyze_game(args.video, args.side, roster, settings)
        out_dir = Path("out") / "json"
        out_dir.mkdir(parents=True, exist_ok=True)
        for play in analyses:
            out_path = out_dir / f"play_{play.play_index}.json"
            out_path.write_text(json.dumps(play, default=lambda o: o.__dict__, indent=2))
        # save a marker that analysis completed
        Path("out/analysis.done").write_text("ok")

        # Optional exports triggered via flags to mirror the command line
        if args.export_summary:
            export_coach_summary(analyses)
        if args.export_clips:
            for pid in roster.keys():
                export_player_clips(analyses, pid)
        if args.export_highlights:
            export_highlights(analyses)
    elif args.cmd == "export":
        # Load analyses from prior run if available
        analyses = []
        if Path("out/json").exists():
            for jf in sorted(Path("out/json").glob("play_*.json")):
                data = load_json_safe(jf, default={})
                analyses.append(
                    analyze_play_from_dict(data)
                )
        if args.report == "coaches":
            export_coach_summary(analyses)
        for pid in args.players:
            export_player_clips(analyses, pid)
        if args.highlights:
            export_highlights(analyses)


def analyze_play_from_dict(data: dict):
    from .models import PlayerGrade, PlayAnalysis

    play = PlayAnalysis(
        play_index=data["play_index"],
        formation=data["formation"],
        play_call=data["play_call"],
        confidence=data["confidence"],
    )
    for pid, g in data.get("assignments", {}).items():
        play.assignments[pid] = PlayerGrade(**g)
    return play


if __name__ == "__main__":  # pragma: no cover
    main()
