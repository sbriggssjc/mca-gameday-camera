def infer_outcome(ball_track) -> dict:
    # Basic heuristic: yards gained = final_x - snap_x in field coords
    # If no field mapping, use pixel delta as proxy and store "unknown_yards"
    return {"result": "unknown", "yards": None}
