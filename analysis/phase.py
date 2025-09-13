import json, pathlib


def _heuristic_phase(row):
    # Heuristic: long ball-flight + everyone runs one way ⇒ likely special teams.
    # Fallback to 'offense/defense' by rp tag if present.
    rp = str(row.get("rp", "unknown"))
    if row.get("avg_ball_speed", 0) > 6.0 and row.get("team_spread", 0) > 0.55:
        return "special_teams", 0.7
    if rp in ("run", "pass"):
        return "offense", 0.6
    return "unknown", 0.4


def _load_model():
    p = pathlib.Path("models/phase_classifier/latest.pt")
    return p if p.exists() else None


def apply(out_dir: str | pathlib.Path = "output"):
    out = pathlib.Path(out_dir)
    p = out/"plays.jsonl"
    lines = [json.loads(x) if x.strip().startswith("{") else x
             for x in p.read_text().splitlines()]
    model_path = _load_model()
    new = []
    for line in lines:
        if isinstance(line, dict):
            if model_path:
                # TODO: replace with real model inference; for now mark unknown w/ low conf
                ph, conf = "unknown", 0.40
            else:
                ph, conf = _heuristic_phase(line)
            line["phase"] = ph
            line["phase_conf"] = round(conf, 2)
            new.append(json.dumps(line))
        else:
            new.append(line)
    p.write_text("\n".join(new))
    print("[phase] updated plays.jsonl")


if __name__ == "__main__":
    import sys
    apply(sys.argv[1] if len(sys.argv) > 1 else "output")

