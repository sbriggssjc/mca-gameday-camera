# analysis/config.py
# Canonical defaults and profiles shared across the analysis package.

DEFAULT_MIN_PLAY_GAP: float = 1.5      # seconds between plays to split
DEFAULT_MIN_PLAY_LEN: float = 6.0      # minimum play duration

PROFILE_DEFAULTS = {
    "game": {
        "min_play_gap": DEFAULT_MIN_PLAY_GAP,
        "min_play_length": DEFAULT_MIN_PLAY_LEN,
        "generate_report": True,
        "generate_clips": True,
        "generate_highlights": True,
        "make_overlay": True,
    },
    "practice": {
        "min_play_gap": 1.2,
        "min_play_length": 4.5,
        "generate_report": False,
        "generate_clips": True,
        "generate_highlights": False,
        "make_overlay": False,
    },
    "clinic": {
        "min_play_gap": 1.0,
        "min_play_length": 4.0,
        "generate_report": False,
        "generate_clips": False,
        "generate_highlights": False,
        "make_overlay": False,
    },
}
