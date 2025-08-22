from pathlib import Path
import statistics as stats


def print_debug_summary(
    out_dir: Path,
    plays,
    predictions,
    grades,
    profile=None,
    min_len=None,
    min_gap=None,
):
    total_plays = len(plays)
    total_fallback = sum(1 for p in plays if p.get("source", "").startswith("fallback"))
    pred_counts = {p: c for p, c in __import__("collections").Counter(
        (r.get("predicted_play") or "UNKNOWN" for r in predictions)
    ).items()}
    known_preds = sum(v for k, v in pred_counts.items() if k != "UNKNOWN")
    unknown_preds = pred_counts.get("UNKNOWN", 0)
    top_match_rate = (known_preds / total_plays) if total_plays else 0.0
    confs = [
        float(r.get("confidence") or 0.0)
        for r in predictions
        if isinstance(r.get("confidence"), (int, float))
    ]
    median_conf = stats.median(confs) if confs else 0.0

    gvals = [
        g.get("overall_defense")
        for g in grades
        if isinstance(g.get("overall_defense"), (int, float))
    ]
    ungradables = total_plays - len(gvals)
    avg_grade = (sum(gvals) / len(gvals)) if gvals else None

    formations = [
        (p.get("formation") or p.get("topk", [{}])[0].get("formation"))
        for p in predictions
    ]
    unknown_form = sum(1 for f in formations if not f or str(f).lower().startswith("unknown"))

    print("\n==== Debug Summary ====")
    print(f"Output dir: {out_dir}")
    if profile or min_len or min_gap:
        print(f"Profile: {profile} | min_play_length={min_len}s | min_play_gap={min_gap}s")
    print(f"Plays detected: {total_plays} (fallback: {total_fallback})")
    print(f"Predicted (known): {known_preds} | UNKNOWN: {unknown_preds}")
    print(f"Top plays: {sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
    print(f"Median confidence: {median_conf:.2f}")
    print(f"Unknown formations: {unknown_form}")
    print(f"Playbook/known rate: {top_match_rate:.2f}")
    if total_plays and (ungradables / total_plays) > 0.4:
        print("Avg defensive grade: ⚠️  insufficient data")
    elif avg_grade is not None:
        print(f"Avg defensive grade: {avg_grade:.2f}")
    else:
        print("Avg defensive grade: N/A")
    print("=======================\n")
