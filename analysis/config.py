DEFAULT_MIN_PLAY_GAP = 1.5
DEFAULT_MIN_PLAY_LEN = 6.0

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
