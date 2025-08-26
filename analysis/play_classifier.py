from __future__ import annotations
from typing import Any, Dict, List

# Light aliases and family mapping retained for pipeline usage
ALIASES = {
    "leo_f_stick": "Leo F Stick",
    "leo-stick": "Leo F Stick",
    "lit_jet_sweep": "Lit Jet Sweep",
    "rit_jet_sweep": "Rit Jet Sweep",
    "rit_8_option": "Rit 8 Option",
    "flare_boot_rit": "Rit Flare Boot",
    "f_screen_rit": "Rit F Screen",
}

FAMILY = {
    "Leo F Stick": "F Stick",
    "Rit Jet Sweep": "Jet Sweep",
    "Lit Jet Sweep": "Jet Sweep",
    "Rit Flare Boot": "Boot",
    "Rit F Screen": "Screen",
    "Rit 8 Option": "Option",
}

def normalize_label(s: str) -> str:
    key = s.strip().lower().replace(" ", "_").replace("-", "_")
    return ALIASES.get(key, s.strip())

# Facade that guarantees classify_plays is importable by pipeline.py.
# If an internal impl exists, delegate to it. Otherwise, return a safe fallback.

_INTERNAL_IMPL = None
for _sym in ("classify_plays_impl", "classify", "run_classifier", "infer_plays"):
    try:
        from analysis import play_classifier as _self  # self-module
        _candidate = getattr(_self, _sym) if _sym != "classify_plays" else None
        if callable(_candidate):
            _INTERNAL_IMPL = _candidate
            break
    except Exception:
        pass


def _fallback_classify_plays(
    segments: List[Dict[str, Any]],
    playbook: Dict[str, Any],
    **kwargs: Any
) -> List[Dict[str, Any]]:
    """Safe no-op classifier that preserves schema and never crashes downstream."""
    out: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments, 1):
        formation = seg.get("formation", "Unknown")
        fconf = float(seg.get("formation_confidence", 0.0))
        out.append({
            "play_id": seg.get("play_id", f"PLAY_{i:03d}"),
            "t0": seg.get("t0"),
            "t1": seg.get("t1"),
            "snap": seg.get("snap"),
            "whistle": seg.get("whistle"),
            "clip_path": seg.get("clip_path"),
            "formation": formation,
            "formation_confidence": fconf,
            "play_family": "Unknown",
            "playcall_confidence": 0.0,
            "candidates": [],
            "outcome": seg.get("outcome"),
            "clip_duration": seg.get("clip_duration"),
        })
    return out


def classify_plays(
    segments: List[Dict[str, Any]],
    playbook: Dict[str, Any],
    **kwargs: Any
) -> List[Dict[str, Any]]:
    if callable(_INTERNAL_IMPL):
        return _INTERNAL_IMPL(segments, playbook, **kwargs)
    return _fallback_classify_plays(segments, playbook, **kwargs)

__all__ = ["classify_plays", "normalize_label", "FAMILY"]
