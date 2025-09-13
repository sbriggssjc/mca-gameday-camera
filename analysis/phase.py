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
    """Return a loaded phase classifier if available.

    The real model implementation is still a work in progress, but we try to
    eagerly load the checkpoint so that downstream code can switch over to
    using it seamlessly once a model exists.  If loading fails for any reason
    (missing file, missing torch dependency, corrupt checkpoint, ...), ``None``
    is returned and heuristics will be used instead.
    """

    p = pathlib.Path("models/phase_classifier/latest.pt")
    if not p.exists():
        return None
    try:  # pragma: no cover - optional dependency
        import torch

        return torch.jit.load(str(p))
    except Exception:
        return None


def apply(out_dir: str | pathlib.Path = "output"):
    out = pathlib.Path(out_dir)
    p = out/"plays.jsonl"
    lines = [json.loads(x) if x.strip().startswith("{") else x
             for x in p.read_text().splitlines()]
    model = _load_model()
    new = []
    for line in lines:
        if isinstance(line, dict):
            if model is None:
                ph, conf = _heuristic_phase(line)
            else:
                # TODO: replace with real model inference once available;
                # for now, model presence only influences the confidence.
                ph, conf = "unknown", 0.40
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

