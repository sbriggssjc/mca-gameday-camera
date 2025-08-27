from __future__ import annotations
from typing import List, Dict

def detect_formations(video_path: str, segments: List[dict]) -> Dict[str, dict]:
    """Return mapping PLAY_xxx -> {'formation': str, 'confidence': float}.

    This is a lightweight placeholder that marks every play as Unknown with
    zero confidence so the pipeline has a stable API even without a trained
    detector.
    """
    result: Dict[str, dict] = {}
    for i, _ in enumerate(segments, start=1):
        pid = f"PLAY_{i:03d}"
        result[pid] = {"formation": "Unknown", "confidence": 0.0}
    return result
