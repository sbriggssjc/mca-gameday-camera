from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence
import json
from tools.json_io import load_json_safe

from .segmentation import Segment


DEFAULT_WEIGHTS = {"contain": 0.35, "interior": 0.35, "coverage": 0.30}


def _load_weights(path: str | None) -> Dict[str, float]:
    if path:
        data = load_json_safe(Path(path))
        if isinstance(data, dict):
            return data  # type: ignore[return-value]
    return DEFAULT_WEIGHTS


def _is_fallback(segments: Sequence[Segment]) -> bool:
    if not segments:
        return False
    durations = [round(seg.duration) for seg in segments]
    return len(set(durations)) == 1 and 10 <= durations[0] <= 14


def grade_plays(
    segments: Sequence[Segment],
    frames: Sequence[object],
    fps: float,
    out_dir: str | Path,
    formations: Sequence[str] | None = None,
    grading_weights: str | None = None,
) -> List[Dict[str, object]]:
    """Generate defensive grades for ``segments``.

    When the segmentation falls back to uniform windows, limited information is
    available.  In that case we emit coarse metrics and mark the mode as
    ``"fallback"`` to ensure the report remains populated.  Otherwise static
    position grades are returned (placeholder for real grading logic).
    """

    weights = _load_weights(grading_weights)
    out_dir = Path(out_dir)
    grades_path = out_dir / "grades.jsonl"

    fallback = _is_fallback(segments)

    results: List[Dict[str, object]] = []
    with grades_path.open("w", encoding="utf8") as f:
        for idx, _seg in enumerate(segments):
            if fallback:
                metrics = {
                    "edge_contain": 0.5,
                    "interior_push": 0.5,
                    "pursuit": 0.5,
                }
                overall = sum(metrics.values()) / len(metrics)
                row = {
                    "play_index": idx,
                    "grading_mode": "fallback",
                    "formation": formations[idx] if formations and idx < len(formations) else "Unknown",
                    "metrics": metrics,
                    "overall_defense": round(overall, 2),
                    "weights": weights,
                }
            else:
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
                    "grading_mode": "standard",
                    "formation": formations[idx] if formations and idx < len(formations) else "Unknown",
                    "defense": defense,
                    "overall_defense": round(overall, 2),
                    "weights": weights,
                }
            results.append(row)
            f.write(json.dumps(row) + "\n")
    return results
