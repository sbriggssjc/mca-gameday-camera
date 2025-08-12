from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from .segmentation import Segment


def match_all(
    segments: Sequence[Segment],
    frames: Sequence[Any],
    fps: float,
    playbook: object | None = None,
) -> List[Dict[str, object]]:
    """Return a best-effort play match for each segment."""

    return [_match_one(seg, frames, fps, playbook) for seg in segments]


def _match_one(
    seg: Segment, frames: Sequence[Any], fps: float, playbook: object | None = None
) -> Dict[str, object]:
    """Heuristic play matcher.

    Motion orientation during the first few seconds of the play is analysed to
    differentiate basic runs such as "Dive" (north-south burst) and "Sweep"
    (lateral flow).  All other plays are reported as ``"Unknown"``.
    """

    start = int(seg.start_ts * fps)
    end = min(len(frames), start + int(4 * fps))
    if end - start <= 1:
        return {"name": "Unknown", "confidence": 0.0}

    horiz = 0.0
    vert = 0.0
    for i in range(start + 1, end):
        prev, cur = frames[i - 1], frames[i]
        if prev is None or cur is None:
            continue
        diff = np.abs(cur.astype("float32") - prev.astype("float32"))
        horiz += float(np.abs(diff[:, 1:] - diff[:, :-1]).sum())
        vert += float(np.abs(diff[1:, :] - diff[:-1, :]).sum())

    total = horiz + vert
    if total == 0:
        return {"name": "Unknown", "confidence": 0.0}

    if horiz > vert * 1.2:
        name = "Sweep"
        conf = horiz / total
    elif vert > horiz * 1.2:
        name = "Dive"
        conf = vert / total
    else:
        name = "Unknown"
        conf = 0.0

    return {"name": name, "confidence": round(conf, 2)}
