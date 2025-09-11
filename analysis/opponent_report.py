from __future__ import annotations
import json, pathlib, csv, statistics, sys

MCA_PLAYS = {
    # Offense suggestions we can call vs THEIR defense
    "edge_fast": ["Rit Jet Sweep", "Leo Jet Sweep", "Reo Quick Screen"],
    "counter_cutback": ["Rit F Counter", "Lit F Counter", "Rit Power R"],
    "flood_vs_zone": ["Reo Flood", "Leo Flood", "F Stick"],
    "boot_vs_flow": ["Rit Flare Boot", "Lit Flare Boot"],
}


def load_jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def load_csv_counts(path):
    counts = {}
    if not path.exists():
        return counts
    with path.open() as f:
        for row in csv.DictReader(f):
            counts.setdefault(row["metric"], {})[row["key"]] = float(row["value"])
    return counts


def top_dirs(counts):
    d = counts.get("direction", {})
    items = sorted(d.items(), key=lambda kv: -kv[1])
    return items[:2]


def infer_def_weakness(def_counts):
    # Heuristic: if direction bias exists, edge is weak to that side
    dir_bias = top_dirs(def_counts)
    recs = []
    if dir_bias and dir_bias[0][0] in ("left", "right") and dir_bias[0][1] >= 6:
        recs.append("edge_fast")
    # If outside_run shows up as successful outcome (requires outcomes counter)
    fam = def_counts.get("family", {})
    if fam.get("outside_run", 0) >= fam.get("inside_run", 0) and fam.get("outside_run", 0) >= 6:
        recs.append("counter_cutback")
    # Default pass game ensures stress on flats/curl
    recs.append("flood_vs_zone")
    recs.append("boot_vs_flow")
    return recs


def positive_outcome_ratio(plays):
    pos = sum(1 for p in plays if p.get("auto_outcome") == "positive")
    return (pos, len(plays))


def shortlist_star_clips(plays, k=6):
    # Use flow p95 (saved under auto_flow.mag_p95) as proxy for impactful gain
    scored = []
    for p in plays:
        m = (p.get("auto_flow") or {}).get("mag_p95", 0.0)
        scored.append((m, pathlib.Path(p.get("src", "")).name))
    scored.sort(reverse=True)
    return [name for _, name in scored[:k]]


def main(out_dir_str):
    out = pathlib.Path(out_dir_str)
    plays = load_jsonl(out / "plays.jsonl")
    # Split sides
    O = [p for p in plays if p.get("lincoln_side") == "offense"]
    D = [p for p in plays if p.get("lincoln_side") == "defense"]

    off_counts = load_csv_counts(out / "tendencies_offense.csv")
    def_counts = load_csv_counts(out / "tendencies_defense.csv")

    # Simple directional bias and family mix
    star_off = shortlist_star_clips(O, k=6)

    # Weakness recs: what WE should call on offense vs THEIR defense
    rec_keys = infer_def_weakness(def_counts)
    suggestions = [item for key in rec_keys for item in MCA_PLAYS.get(key, [])]

    # Quick summary lines
    o_pos, o_tot = positive_outcome_ratio(O)
    d_pos, d_tot = positive_outcome_ratio(D)
    lines = []
    lines.append("# Lincoln Christian — Opponent Report\n")
    lines.append(
        f"**Clips analyzed:** {len(plays)}  |  **Lincoln Offense:** {len(O)}  |  **Lincoln Defense:** {len(D)}\n"
    )
    lines.append("## Lincoln Offense Tendencies (quick peek)")
    rp = off_counts.get("run_pass", {})
    fam = off_counts.get("family", {})
    lines.append(f"- Run/Pass (auto): {int(rp.get('run', 0))} run / {int(rp.get('pass', 0))} pass")
    if fam:
        lines.append(
            f"- Families top: {sorted(fam.items(), key=lambda kv: -kv[1])[:4]}"
        )
    lines.append(f"- Positive outcome rate (auto): {o_pos}/{o_tot}\n")
    lines.append("**Star clips to review (highest impact motion):**")
    for n in star_off:
        lines.append(f"- {n}")
    lines.append("\n## Lincoln Defense Tendencies (quick peek)")
    rpD = def_counts.get("run_pass", {})
    famD = def_counts.get("family", {})
    lines.append(
        f"- Offense faced (for context): {int(rpD.get('run', 0))} runs / {int(rpD.get('pass', 0))} passes"
    )
    if famD:
        lines.append(
            f"- Offense types vs them: {sorted(famD.items(), key=lambda kv:-kv[1])[:4]}"
        )
    lines.append(
        f"- Positive outcome rate allowed (auto proxy): {d_pos}/{d_tot}\n"
    )

    lines.append("## Suggested MCA Calls vs Lincoln Defense")
    for s in suggestions:
        lines.append(f"- **{s}**")
    lines.append(
        "\n_Notes_: suggestions are heuristic. Confirm with the star clips and your assistants. If you tag 10–15 clips with `quick_tag`, rerun tendencies for sharper recs."
    )
    (out / "opponent_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("[opponent_report] wrote", out / "opponent_report.md")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output")
