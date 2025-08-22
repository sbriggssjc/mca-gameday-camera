"""Generate a very small coach summary report.

This module intentionally keeps dependencies light by avoiding heavy PDF
libraries.  The :func:`generate` function writes a CSV table summarising
plays and a plain text file with a ``.pdf`` extension acting as a stand in
for a real PDF.  The aim is to exercise the file layout and allow unit
tests to verify that the pipeline invoked the reporting stage.
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, Dict


def generate(plays: Iterable[Dict[str, object]], grades: Iterable[Dict[str, object]], out_dir: str) -> None:
    """Write ``coach_summary.csv`` and ``coach_summary.pdf`` into ``out_dir``.

    Parameters
    ----------
    plays, grades:
        Iterables of dictionaries.  Only a handful of keys are used to produce
        the summary table.
    out_dir:
        Base output directory.  A ``reports`` sub directory is created within
        this folder.
    """

    reports_dir = os.path.join(out_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    csv_path = os.path.join(reports_dir, "coach_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["play_id", "predicted_play"])
        for play in plays:
            writer.writerow([play.get("play_id"), play.get("predicted_play")])

    pdf_path = os.path.join(reports_dir, "coach_summary.pdf")
    with open(pdf_path, "w", encoding="utf8") as f:
        f.write("Coach summary placeholder\n")
        for play in plays:
            f.write(f"Play {play.get('play_id')}: {play.get('predicted_play')}\n")
