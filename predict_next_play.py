from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import List, Dict, Union

CSV_PATH = Path("scouting_data.csv")


def _parse_int(val: str | int | None) -> int | None:
    try:
        return int(str(val).strip().lstrip("Q"))
    except Exception:
        return None


def _norm_form(form: str | None) -> str:
    return (form or "").lower().replace(" ", "").replace("-", "")


def _similar_form(a: str | None, b: str | None) -> bool:
    fa = _norm_form(a)
    fb = _norm_form(b)
    return fa == fb or fa in fb or fb in fa


def predict_play(
    opponent: str,
    formation: str | None,
    down: int | None,
    distance: int | None,
    quarter: int | None,
) -> List[Dict[str, int]] | str:
    """Predict next play from scouting data.

    Returns top two play labels with percentages or a message string if data is
    insufficient.
    """

    if not CSV_PATH.exists():
        return "No prediction available"

    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("opponent", "").lower() != opponent.lower():
                continue
            if formation and row.get("formation") and not _similar_form(
                row["formation"], formation
            ):
                continue
            r_down = _parse_int(row.get("down"))
            if down is not None and r_down is not None and r_down != down:
                continue
            r_qtr = _parse_int(row.get("quarter"))
            if quarter is not None and r_qtr is not None and abs(r_qtr - quarter) > 1:
                continue
            r_dist = _parse_int(row.get("distance") or row.get("yards_to_go"))
            if distance is not None and r_dist is not None and abs(r_dist - distance) > 2:
                continue
            rows.append(row)

    if len(rows) < 3:
        return "No prediction available"

    counts = Counter(r.get("label", "Unknown") for r in rows)
    total = sum(counts.values())
    preds: List[Dict[str, int]] = []
    for label, cnt in counts.most_common(2):
        pct = int(round((cnt / total) * 100)) if total else 0
        preds.append({"label": label, "percent": pct})
    return preds


__all__ = ["predict_play"]
