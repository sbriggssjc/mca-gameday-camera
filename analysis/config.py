# analysis/config.py
# Central defaults shared across the pipeline

# Segmentation defaults
DEFAULT_MIN_PLAY_GAP: float = 1.5   # seconds between detected plays
DEFAULT_MIN_PLAY_LEN: float = 6.0   # minimum duration of a play

# Profile presets used by pipeline.py via --profile {game,practice,clinic}
# Each profile can override min gaps/lengths or toggle outputs.
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
        "min_play_gap": 1.0,
        "min_play_length": 4.0,
        "generate_report": False,
        "generate_clips": True,
        "generate_highlights": False,
        "make_overlay": False,
    },
    "clinic": {
        "min_play_gap": 0.8,
        "min_play_length": 3.5,
        "generate_report": False,
        "generate_clips": True,
        "generate_highlights": False,
        "make_overlay": False,
    },
}
