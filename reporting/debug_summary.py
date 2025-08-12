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
    known_preds = sum(
        1 for p in predictions if p.get("predicted_play") and p.get("predicted_play") != "UNKNOWN"
    )
    top_match_rate = 0.0
    if total_plays:
        top_match_rate = known_preds / total_plays

    gvals = [g.get("overall_defense") for g in grades if isinstance(g.get("overall_defense"), (int, float))]
    avg_grade = (sum(gvals) / len(gvals)) if gvals else None

    formations = [(p.get("formation") or p.get("topk", [{}])[0].get("formation")) for p in predictions]
    unknown_form = sum(1 for f in formations if not f or str(f).lower().startswith("unknown"))

    print("\n==== Debug Summary ====")
    print(f"Output dir: {out_dir}")
    if profile or min_len or min_gap:
        print(f"Profile: {profile} | min_play_length={min_len}s | min_play_gap={min_gap}s")
    print(f"Plays detected: {total_plays} (fallback: {total_fallback})")
    print(f"Predicted (known): {known_preds} | Unknown formations: {unknown_form}")
    print(f"Playbook/known rate: {top_match_rate:.2f}")
    if avg_grade is not None:
        print(f"Avg defensive grade: {avg_grade:.2f}")
    else:
        print("Avg defensive grade: N/A")
    print("=======================\n")
