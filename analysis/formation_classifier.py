from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np

from .assignments import Playbook
from .segmentation import Segment


def classify_all(
    segments: Sequence[Segment],
    frames: Sequence[Any],
    fps: float,
    playbook: Playbook | None,
) -> List[str]:
    """Classify offensive formation for each ``Segment``.

    The classifier looks at roughly the first 0.8 seconds of each play and
    compares motion on the left and right halves of the frame.  It infers a
    coarse tag such as ``"Rit"`` or ``"Lit"`` and falls back to ``"Unknown"``
    when insufficient motion exists.
    """

    return [_classify_one(seg, frames, fps, playbook) for seg in segments]


def _classify_one(
    seg: Segment, frames: Sequence[Any], fps: float, playbook: Playbook | None
) -> str:
    start = int(seg.start_ts * fps)
    end = min(len(frames), start + int(0.8 * fps))
    if end - start <= 1:
        if playbook and playbook.offense_plays:
            return playbook.offense_plays[0].formation
        return "Unknown"

    left_motion = 0.0
    right_motion = 0.0
    for i in range(start + 1, end):
        prev, cur = frames[i - 1], frames[i]
        if prev is None or cur is None:
            continue
        diff = np.abs(cur.astype("float32") - prev.astype("float32"))
        h, w = diff.shape[:2]
        left_motion += float(diff[:, : w // 2].sum())
        right_motion += float(diff[:, w // 2 :].sum())

    if left_motion == right_motion == 0:
        return "Unknown"

    side = "R" if right_motion >= left_motion else "L"
    balance = abs(right_motion - left_motion) / (right_motion + left_motion)
    if balance < 0.1:
        suffix = "eo"
    elif balance > 0.6:
        suffix = "end"
    else:
        suffix = "it"
    return f"{side}{suffix}"


def classify_formation(playbook: Playbook | None, frames: Sequence[Any], fps: float) -> str:
    """Compatibility wrapper returning a single formation for tests."""

    res = classify_all([Segment(0, 1)], frames, fps, playbook)
    return res[0] if res else "Unknown"
