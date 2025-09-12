from __future__ import annotations
import json, pathlib, csv, sys, collections

MCA_PLAYS = {
    # Offense suggestions we can call vs THEIR defense
    "edge_fast": ["Rit Jet Sweep", "Leo Jet Sweep", "Reo Quick Screen"],
    "counter_cutback": ["Rit F Counter", "Lit F Counter", "Rit Power R"],
    "flood_vs_zone": ["Reo Flood", "Leo Flood", "F Stick"],
    "boot_vs_flow": ["Rit Flare Boot", "Lit Flare Boot"],
}


def load_jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def try_load_csv(path: pathlib.Path):
    """Return list of rows from CSV if path exists, else None."""
    p = pathlib.Path(path)
    if not p.exists():
        return None
    with p.open() as f:
        return list(csv.DictReader(f))


def summarize(csv_rows):
    """Summarize tendencies CSV into convenient counters."""
    rp = collections.Counter()
    fam = collections.Counter()
    direction = collections.Counter()
    pos = 0
    total = 0
    for r in csv_rows or []:
        metric = r.get("metric")
        val = r.get("value")
        c = int(r.get("count") or r.get("value") or 0)
        if metric == "rp":
            rp[val] += c
        elif metric == "family":
            fam[val] += c
        elif metric == "direction":
            direction[val] += c
        elif metric == "outcome" and val == "positive":
            pos += c
        elif metric == "total" and val in ("clips", "plays"):
            total = c
    return rp, fam, direction, pos, total


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


def main():
    out_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    out = pathlib.Path(out_arg) if out_arg else pathlib.Path("output")

    plays_path = out / "plays.jsonl"
    if not plays_path.exists():
        raise SystemExit(
            f"[opponent_report] plays.jsonl not found at '{plays_path}'. "
            "Tip: verify OUT matches your pipeline run (e.g., OUT=output/opponent_lincoln_20250912) "
            "and call: python -m analysis.opponent_report \"$OUT\""
        )

    plays = load_jsonl(plays_path)
    # Split sides
    O = [p for p in plays if p.get("lincoln_side") == "offense"]
    D = [p for p in plays if p.get("lincoln_side") == "defense"]

    off_csv = try_load_csv(out / "tendencies_offense.csv")
    def_csv = try_load_csv(out / "tendencies_defense.csv")

    off_rp, off_fam, off_dir, off_pos_csv, off_total_csv = summarize(off_csv)
    def_rp, def_fam, def_dir, def_pos_csv, def_total_csv = summarize(def_csv)

    o_pos, o_tot = (
        (off_pos_csv, off_total_csv) if off_total_csv else positive_outcome_ratio(O)
    )
    d_pos, d_tot = (
        (def_pos_csv, def_total_csv) if def_total_csv else positive_outcome_ratio(D)
    )

    star_off = shortlist_star_clips(O, k=6)

    def_counts = {"direction": def_dir, "family": def_fam}
    rec_keys = infer_def_weakness(def_counts)
    suggestions = [item for key in rec_keys for item in MCA_PLAYS.get(key, [])]

    lines = []
    lines.append("# Lincoln Christian — Opponent Report\n")
    lines.append(
        f"**Clips analyzed:** {len(plays)}  |  **Lincoln Offense:** {off_total_csv or len(O)}  |  **Lincoln Defense:** {def_total_csv or len(D)}\n"
    )
    lines.append("## Lincoln Offense Tendencies (quick peek)")
    lines.append(
        f"- Run/Pass (auto): {int(off_rp.get('run', 0))} run / {int(off_rp.get('pass', 0))} pass"
    )
    if off_fam:
        lines.append(f"- Families top: {off_fam.most_common(5)}")
    lines.append(f"- Positive outcome rate (auto): {o_pos}/{o_tot}\n")
    lines.append("**Star clips to review (highest impact motion):**")
    for n in star_off:
        lines.append(f"- {n}")
    lines.append("\n## Lincoln Defense Tendencies (quick peek)")
    lines.append(
        f"- Offense faced (for context): {int(def_rp.get('run', 0))} runs / {int(def_rp.get('pass', 0))} passes"
    )
    if def_fam:
        lines.append(f"- Offense types vs them: {def_fam.most_common(5)}")
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
    main()
