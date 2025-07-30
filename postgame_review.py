from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from play_classifier import classify_play


CORRECTIONS_LOG = Path("training/logs/postgame_corrections.json")


def rerun_play_analysis(folder: str, model: str = "models/play_classifier/latest.pt") -> List[Dict[str, Any]]:
    """Re-analyze clips in ``folder`` with the latest model."""
    clips = sorted(Path(folder).rglob("*.mp4"))
    if not clips:
        print(f"[!] No clips found in {folder}")
        return []

    results: List[Dict[str, Any]] = []
    corrections: List[Dict[str, Any]] = []

    for clip in clips:
        meta_path = clip.with_suffix(".json")
        old_label = "unknown"
        metadata: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                old_label = str(metadata.get("play_type", "unknown"))
            except Exception:
                metadata = {}
        prediction = classify_play(str(clip), str(meta_path) if meta_path.exists() else None, model)
        new_label = prediction["play_type"]
        conf = float(prediction["confidence"])
        results.append({
            "clip": clip.name,
            "play_type": new_label,
            "confidence": conf,
            "players": metadata.get("players", []),
            "formation": metadata.get("formation"),
            "alignment": metadata.get("alignment"),
            "missed_assignment": metadata.get("missed_assignment"),
            "success": metadata.get("success"),
        })
        if new_label != old_label and conf > 0.9:
            metadata["play_type"] = new_label
            metadata["confidence"] = conf
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            corrections.append({
                "clip": clip.name,
                "original": old_label,
                "reclassified": new_label,
                "confidence": conf,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })

    if corrections:
        CORRECTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            with CORRECTIONS_LOG.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
        data.extend(corrections)
        with CORRECTIONS_LOG.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    return results


def generate_performance_summary(plays: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return per-player performance stats from ``plays``."""
    players: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "snap_count": 0,
        "alignments": Counter(),
        "play_types": Counter(),
        "successes": 0,
        "failures": 0,
    })

    for p in plays:
        for pid in p.get("players", []):
            info = players[str(pid)]
            info["snap_count"] += 1
            if p.get("alignment"):
                info["alignments"][p["alignment"]] += 1
            if p.get("play_type"):
                info["play_types"][p["play_type"]] += 1
            if p.get("success") is True:
                info["successes"] += 1
            elif p.get("success") is False:
                info["failures"] += 1

    summary: Dict[str, Any] = {}
    for pid, data in players.items():
        total = data["successes"] + data["failures"]
        rate = data["successes"] / total if total else None
        summary[pid] = {
            "snap_count": data["snap_count"],
            "most_common_alignment": data["alignments"].most_common(1)[0][0] if data["alignments"] else None,
            "most_used_play_types": [pt for pt, _ in data["play_types"].most_common(3)],
            "success_rate": rate,
        }
    return summary


def suggest_practice_focus(plays: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    """Return weekly practice recommendations."""
    missed_counter: Counter[str] = Counter()
    formation_counter: Counter[str] = Counter()
    for p in plays:
        if p.get("missed_assignment"):
            missed_counter[p["missed_assignment"]] += 1
        if p.get("formation"):
            formation_counter[p["formation"]] += 1

    total_plays = len(plays) or 1
    overused = [f for f, cnt in formation_counter.items() if cnt / total_plays > 0.4]

    suggestions: List[str] = []
    if overused:
        suggestions.append(f"Consider mixing in other formations instead of {'/'.join(overused)}")
    top_missed = [a for a, _ in missed_counter.most_common(3)]
    for miss in top_missed:
        suggestions.append(f"Practice {miss} assignments")

    return {
        "top_missed_assignments": top_missed,
        "overused_formations": overused,
        "suggestions": suggestions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Postgame self-learning review")
    parser.add_argument("--review", default="last_game", help="Folder with game clips")
    args = parser.parse_args()

    plays = rerun_play_analysis(args.review)
    if not plays:
        return
    summary = generate_performance_summary(plays)
    focus = suggest_practice_focus(plays, summary)

    Path(args.review).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.review) / "postgame_summary.json"
    focus_path = Path(args.review) / "practice_focus.md"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"players": summary, "total_plays": len(plays)}, f, indent=2)

    lines = ["# Practice Focus"]
    for s in focus["suggestions"]:
        lines.append(f"- {s}")
    with focus_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\u2705 Summary saved to {summary_path}")
    print(f"\u2705 Practice focus saved to {focus_path}")


if __name__ == "__main__":
    main()
