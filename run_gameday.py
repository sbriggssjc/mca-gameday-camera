#!/usr/bin/env python3
"""Master pipeline runner for game day automation."""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    """Run ``cmd`` and raise ``CalledProcessError`` on failure."""
    logging.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _send_webhook(msg: str) -> None:
    """Post ``msg`` to WEBHOOK_URL if defined."""
    url = os.environ.get("WEBHOOK_URL")
    if not url:
        return
    try:
        import requests  # type: ignore

        requests.post(url, json={"text": msg}, timeout=10)
    except Exception as exc:  # pragma: no cover - optional network
        logging.warning("Webhook failed: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full game day pipeline")
    parser.add_argument("--opponent", required=True, help="Opponent name")
    parser.add_argument(
        "--game_date",
        default=_dt.date.today().isoformat(),
        help="Game date YYYY-MM-DD",
    )
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard")
    parser.add_argument("--hudl", action="store_true", help="Generate HUDL export")
    parser.add_argument("--email", action="store_true", help="Send postgame email")
    parser.add_argument("--summary", action="store_true", help="Generate reports")
    parser.add_argument("--relearn", action="store_true", help="Reclassify clips")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"gameday_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    start = _dt.datetime.now()
    logging.info("Gameday run started for %s (%s)", args.opponent, args.game_date)

    try:
        _run([sys.executable, "stream_to_youtube.py"])
        _run([sys.executable, "ai_detector.py"])

        if args.summary:
            _run([sys.executable, "generate_analysis_summary.py"])
            _run([sys.executable, "generate_scouting_report.py", args.opponent])
            _run([sys.executable, "generate_install_plan.py", "--opponent", args.opponent])

        if args.hudl:
            _run([sys.executable, "generate_hudl_csv.py", "--opponent", args.opponent])

        if args.relearn:
            _run([sys.executable, "reclassify_old_clips.py"])

        if args.email:
            _run([sys.executable, "send_postgame_email.py", args.opponent, "0", "0", "", ""])

        if args.dashboard:
            _run([sys.executable, "dashboard.py"])
    finally:
        elapsed = (_dt.datetime.now() - start).total_seconds()
        logging.info("Gameday run finished in %.1f seconds", elapsed)
        _send_webhook(f"Gameday pipeline finished in {elapsed:.1f}s for {args.opponent}")


if __name__ == "__main__":
    main()

