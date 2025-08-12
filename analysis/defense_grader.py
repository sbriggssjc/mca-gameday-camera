from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from .segmentation import Segment


DEFAULT_WEIGHTS = {"contain": 0.35, "interior": 0.35, "coverage": 0.30}


def _load_weights(path: str | None) -> Dict[str, float]:
    if path and Path(path).exists():
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            pass
    return DEFAULT_WEIGHTS


def grade_plays(
    segments: Sequence[Segment],
    formations: Sequence[str],
    out_dir: str | Path,
    grading_weights: str | None = None,
) -> List[Dict[str, object]]:
    """Generate very simple defensive grades for ``segments``.

    Each play receives static position scores.  The results are written to
    ``grades.jsonl`` within ``out_dir`` and also returned as a list for further
    processing.
    """

    weights = _load_weights(grading_weights)
    out_dir = Path(out_dir)
    grades_path = out_dir / "grades.jsonl"

    results: List[Dict[str, object]] = []
    with grades_path.open("w", encoding="utf8") as f:
        for idx, _seg in enumerate(segments):
            defense = {
                "LE": 0.8,
                "RE": 0.8,
                "DT1": 0.8,
                "DT3": 0.8,
                "Mike": 0.8,
                "Will": 0.8,
                "Monster": 0.8,
                "Blood": 0.8,
                "RCB": 0.8,
                "LCB": 0.8,
                "FS": 0.8,
            }
            overall = sum(defense.values()) / len(defense)
            row = {
                "play_index": idx,
                "formation": formations[idx] if idx < len(formations) else "Unknown",
                "defense": defense,
                "overall_defense": round(overall, 2),
                "weights": weights,
            }
            results.append(row)
            f.write(json.dumps(row) + "\n")
    return results
