#!/usr/bin/env python3
"""Run the full analysis pipeline end-to-end in one command."""
from __future__ import annotations

import argparse
import json
from tools.json_io import load_json_safe
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np


log = logging.getLogger(__name__)


def run(cmd: List[str], log_file: Path) -> subprocess.CompletedProcess:
    """Run ``cmd`` teeing output to terminal and ``log_file``."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
        proc.wait()
        return subprocess.CompletedProcess(cmd, proc.returncode)


def render_pdf_reportlab(out_dir: Path) -> Path:
    """Render ``report.md`` under ``out_dir`` into ``reports/report.pdf`` using ReportLab."""
    md = out_dir / "reports" / "report.md"
    pdf_dir = out_dir / "reports"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / "report.pdf"
    try:
        from reportlab.platypus import Preformatted, SimpleDocTemplate
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
    except Exception as exc:  # pragma: no cover - dependency missing
        log.warning("ReportLab not available: %s", exc)
        return pdf_path
    txt = md.read_text(encoding="utf-8", errors="replace")
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    style = getSampleStyleSheet()["Code"]
    doc.build([Preformatted(txt, style)])
    log.info("ReportLab PDF written: %s", pdf_path)
    return pdf_path


def render_pdf_fpdf(out_dir: Path) -> Path:
    """Render ``report.md`` using FPDF; fall back to ReportLab on error."""
    md = out_dir / "reports" / "report.md"
    pdf_path = out_dir / "reports" / "report.pdf"
    try:
        from fpdf import FPDF
    except Exception as exc:  # pragma: no cover - dependency missing
        log.warning("FPDF not available: %s", exc)
        return render_pdf_reportlab(out_dir)
    txt = md.read_text(encoding="utf-8", errors="replace")
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_left_margin(12)
    pdf.set_right_margin(12)
    pdf.set_font("Helvetica", size=12)
    for line in txt.splitlines():
        pdf.multi_cell(0, 8, line)
    try:
        pdf.output(str(pdf_path))
    except Exception as exc:  # pragma: no cover - backend error
        log.warning("FPDF failed: %s", exc)
        return render_pdf_reportlab(out_dir)
    log.info("FPDF PDF written: %s", pdf_path)
    return pdf_path


def fallback_segments_from_motion(
    video_path: str, threshold: float, min_seg: float
) -> list[tuple[float, float]]:
    """Return motion-based segments from ``video_path``.

    Frames are sampled at roughly 5fps. The mean absolute difference between
    consecutive sampled frames is normalised to ``[0,1]`` and compared against
    ``threshold``. Detected motion spans shorter than ``min_seg`` seconds are
    discarded and neighbouring regions separated by less than ``0.8`` seconds
    are merged.
    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(fps / 5))
    prev = None
    scores: list[float] = []
    times: list[float] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            if prev is not None:
                diff = cv2.absdiff(prev, blur)
                diff = cv2.normalize(
                    diff.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX
                )
                scores.append(float(np.mean(diff)))
                times.append(frame_idx / fps)
            prev = blur
        frame_idx += 1
    cap.release()

    motion = [s > threshold for s in scores]
    segments: list[tuple[float, float]] = []
    start = None
    for t, m in zip(times, motion):
        if m:
            if start is None:
                start = t
        else:
            if start is not None:
                end = t
                if end - start >= min_seg:
                    segments.append((start, end))
                start = None
    if start is not None and times:
        end = times[-1]
        if end - start >= min_seg:
            segments.append((start, end))

    # Snap→Whistle is the single segmentation pass; no merges or re-windowizing allowed.
    merged: list[list[float]] = [(s, e) for s, e in segments]
    return [(s, e) for s, e in merged]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-click film analysis")
    parser.add_argument("--video", required=True, help="MP4 path")
    parser.add_argument("--team", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--playbook", default="playbooks/mca_5th_playbook.json")
    parser.add_argument("--player-ids")
    parser.add_argument("--wins-threshold", type=float, default=3.0)
    parser.add_argument("--corrections-threshold", type=float, default=2.0)
    parser.add_argument("--min-clip-sec", type=float, default=1.5)
    parser.add_argument("--fallback-segmentation", action="store_true")
    parser.add_argument("--min-valid-plays", type=int, default=5)
    parser.add_argument("--motion-threshold", type=float, default=0.45)
    parser.add_argument("--min-seg-sec", type=float, default=2.0)
    parser.add_argument("--pad-before", type=float, default=0.6)
    parser.add_argument("--pad-after", type=float, default=1.2)
    parser.add_argument(
        "--out",
        help="Output directory",
        default=None,
    )
    parser.add_argument(
        "--pdf-engine",
        choices=["reportlab", "fpdf", "none"],
        default="reportlab",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip clip rendering")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.min_clip_sec < 0.5:
        print("--min-clip-sec must be >= 0.5", file=sys.stderr)
        return 1

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except Exception:
        duration = 0.0
    if probe.returncode != 0 or duration <= 0:
        print("ffprobe failed or returned invalid duration", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(
        args.out if args.out else Path("output") / f"{video_path.stem}_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # ------------------------------------------------------------------ pipeline
    pipeline_cmd = [
        sys.executable,
        "-m",
        "analysis.pipeline",
        "--video",
        str(video_path),
        "--team",
        args.team,
        "--playbook",
        args.playbook,
        "--out",
        str(out_dir),
        "--generate-report",
        "--clip-corrections",
        "--clip-wins",
        "--clip-highlights",
    ]
    if args.player_ids:
        pipeline_cmd += ["--player-ids", args.player_ids]
    pipeline_log = out_dir / "pipeline.log"
    res = run(pipeline_cmd, pipeline_log)
    if res.returncode != 0:
        print("pipeline failed", file=sys.stderr)
        try:
            tail = pipeline_log.read_text().splitlines()[-20:]
            print("\n".join(tail), file=sys.stderr)
        except Exception:
            pass
        return 2

    # required artefacts
    required = [
        "plays.jsonl",
        "play_predictions.jsonl",
        "grades.jsonl",
        "tracking.jsonl",
        "metadata.json",
    ]
    missing = [r for r in required if not (out_dir / r).exists()]
    if missing:
        print("Missing artefacts:", ", ".join(missing), file=sys.stderr)
        return 3

    plays_path = out_dir / "plays.jsonl"
    try:
        with plays_path.open("r", encoding="utf-8") as fh:
            raw_play_count = sum(1 for line in fh if line.strip())
    except FileNotFoundError:
        raw_play_count = 0

    plays_after_fallback = raw_play_count
    if (
        args.fallback_segmentation
        and raw_play_count < args.min_valid_plays
    ):
        segments = fallback_segments_from_motion(
            str(video_path), args.motion_threshold, args.min_seg_sec
        )
        padded: list[tuple[float, float]] = []
        for start, end in segments:
            s = max(0.0, start - args.pad_before)
            e = min(duration, end + args.pad_after)
            if e > s:
                padded.append((s, e))
        plays_after_fallback = len(padded)
        lines = [
            json.dumps(
                {
                    "play_id": i + 1,
                    "start_s": s,
                    "end_s": e,
                    "label": "UNKNOWN",
                    "source": "fallback_motion",
                }
            )
            for i, (s, e) in enumerate(padded)
        ]
        plays_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        log_file = out_dir / "logs" / "one_click.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as lf:
            lf.write(
                f"{datetime.now().isoformat()} fallback_segmentation segments={plays_after_fallback}\n"
            )

    # ---------------------------------------------------------------- rich_summary
    rs_cmd = [
        sys.executable,
        "rich_summary.py",
        "--video",
        str(video_path),
        "--out",
        str(out_dir),
        "--team",
        args.team,
        "--opponent",
        args.opponent,
        "--date",
        args.date,
        "--export-hudl",
        "--per-player-cutups",
        "--wins-threshold",
        str(args.wins_threshold),
        "--corrections-threshold",
        str(args.corrections_threshold),
        "--min-clip-sec",
        str(args.min_clip_sec),
        "--pdf-engine",
        "none",
    ]
    if args.dry_run:
        rs_cmd.append("--dry-run")
    rs_log = out_dir / "rich_summary.log"
    rs_res = run(rs_cmd, rs_log)
    report_md = out_dir / "reports" / "report.md"
    dashboard = out_dir / "dashboards" / "summary.json"
    if rs_res.returncode != 0 and not (report_md.exists() and dashboard.exists()):
        print("rich_summary failed", file=sys.stderr)
        return 4

    # ---------------------------------------------------------------- PDF
    render_pdf_reportlab(out_dir)

    # ---------------------------------------------------------------- summary
    summary = load_json_safe(dashboard, default={}) or {}
    wins = summary.get("wins", 0)
    corrections = summary.get("corrections", 0)
    print("Output:", out_dir)
    print(f"Plays detected (raw): {raw_play_count}")
    print(f"Plays after fallback: {plays_after_fallback}")
    print(f"Wins: {wins}, Corrections: {corrections}")
    print("Report:", out_dir / "reports" / "report.pdf")
    print("Team highlights:", out_dir / "clips" / "highlights" / "team_highlights.mp4")
    print("Wins reel:", out_dir / "clips" / "wins" / "top_wins.mp4")
    print("Corrections reel:", out_dir / "clips" / "corrections" / "top_corrections.mp4")
    print("HUDL CSV:", out_dir / "hudl_export" / "hudl_export.csv")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
