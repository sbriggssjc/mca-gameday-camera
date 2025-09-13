"""Compute simple opponent tendencies from ``plays.jsonl``.

This module focuses on the generic scouting data produced by the upgraded
pipeline.  The implementation intentionally favours clarity over performance
and keeps external dependencies to a minimum so it can run in the unit tests.

The public entry point is :func:`run_from_pipeline` which is used by the
pipeline module.  A command line interface is also provided for ad‑hoc runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

Play = Dict[str, Any]

GENERIC_FORMATIONS = {
    # simple mapper if your formation model already outputs something; otherwise leave as-is
    # e.g., normalize “TripsRight”, “Trips R” -> “trips”
}


def norm_formation(val: Any) -> str:
    if not val:
        return "unknown"
    v = str(val).lower().replace(" ", "").replace("_", "")
    for k in GENERIC_FORMATIONS:
        if v == k:
            return GENERIC_FORMATIONS[k]
    if "trips" in v:
        return "trips"
    if "doubles" in v or "2x2" in v:
        return "doubles"
    if "empty" in v:
        return "empty"
    if "tight" in v or "wing" in v:
        return "tight"
    if "i" == v or "i-" in v:
        return "i"
    if "single" in v or "ace" in v:
        return "singleback"
    return v


def norm_run_dir(val: Any) -> str:
    if not val:
        return "unknown"
    v = str(val).lower()
    if "left" in v:
        return "left"
    if "right" in v:
        return "right"
    if "middle" in v or "mid" in v or "center" in v:
        return "middle"
    return "unknown"


def to_pass_family(p: Play) -> str:
    rp = str(p.get("run_pass") or p.get("rp") or "unknown")
    if rp != "pass":
        return "unknown"
    f = str(p.get("family") or "").lower()
    tags = [str(t).lower() for t in (p.get("tags") or [])]
    if any(k in f for k in ["screen", "bubble", "tunnel", "quick"]) or any(k in tags for k in ["screen", "bubble", "tunnel", "quick"]):
        return "screens/quick"
    if any(k in f for k in ["boot", "bootleg", "naked", "waggle"]) or any(k in tags for k in ["boot", "bootleg", "naked", "waggle"]):
        return "boot"
    if any(k in f for k in ["flood", "corner", "post", "go", "seam", "fade", "wheel", "switch", "smash", "levels", "mesh", "verts", "four"]) or any(k in tags for k in ["flood", "corner", "post", "go", "seam", "fade", "wheel", "switch", "smash", "levels", "mesh", "verts", "four"]):
        return "intermediate/deep"
    return "unknown"


def bucket(y: float) -> str:
    y = float(y)
    if y <= -5:
        return "<=-5"
    if -4 <= y <= 0:
        return "-4–0"
    if 1 <= y <= 3:
        return "1–3"
    if 4 <= y <= 7:
        return "4–7"
    if 8 <= y <= 12:
        return "8–12"
    return "13+"



# ---------------------------------------------------------------------------
# Data loading and filtering

def load_plays(out_dir: Path) -> List[Play]:
    path = out_dir / "plays.jsonl"
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def filter_plays(plays: Iterable[Play], *, only_off=False, only_def=False,
                 exclude_phase: Iterable[str] | None = None,
                 min_conf: float | None = None,
                 use_raw_side: bool = False) -> List[Play]:
    exclude_phase = set(exclude_phase or [])
    result = []
    for p in plays:
        if p.get("phase") in exclude_phase:
            continue
        if use_raw_side:
            side = p.get("lincoln_side")
            conf = float(p.get("lincoln_side_conf") or 0)
        else:
            side = p.get("lincoln_side_final") or p.get("lincoln_side")
            conf = float(p.get("lincoln_side_final_conf") or p.get("lincoln_side_conf") or 0)
        if min_conf is not None and side != "unknown" and conf < min_conf:
            continue
        if only_off and side != "offense":
            continue
        if only_def and side != "defense":
            continue
        result.append(p)
    return result


# ---------------------------------------------------------------------------
# Aggregation helpers

def _success(play: Play) -> bool:
    yards = float(play.get("yards_gained") or 0.0)
    rp = play.get("run_pass")
    if rp == "run":
        return yards >= 3.0
    if rp == "pass":
        return yards >= 5.0
    return False


def summarise(plays: Iterable[Play]) -> Dict[str, Dict[str, List[Play]]]:
    agg: Dict[str, Dict[str, List[Play]]] = {
        'run_pass': defaultdict(list),
        'formation_text': defaultdict(list),
        'offense_personnel': defaultdict(list),
        'run_direction': defaultdict(list),
        'route_primary': defaultdict(list),
        'formation': defaultdict(list),
        'run_dir': defaultdict(list),
        'pass_family': defaultdict(list),
        'yards_bucket': defaultdict(list),
    }
    for p in plays:
        for key in [
            'run_pass',
            'formation_text',
            'offense_personnel',
            'run_direction',
            'route_primary',
            'formation',
        ]:
            if key == 'formation':
                val = norm_formation(p.get('formation') or p.get('generic_formation'))
            else:
                val = p.get(key, 'unknown') or 'unknown'
            agg[key][val].append(p)
        rp = p.get('run_pass') or 'unknown'
        if rp == 'run':
            val = norm_run_dir(p.get('run_direction') or p.get('dir'))
            agg['run_dir'][val].append(p)
        elif rp == 'pass':
            pf = to_pass_family(p)
            agg['pass_family'][pf].append(p)
        yards = p.get('yards')
        if yards is not None:
            agg['yards_bucket'][bucket(yards)].append(p)
    return agg

def _metric_rows(agg: Dict[str, Dict[str, List[Play]]]) -> List[Tuple[str, str, int, float, float, float, int]]:
    rows: List[Tuple[str, str, int, float, float, float, int]] = []
    for metric, groups in agg.items():
        for val, plays in groups.items():
            yards = [float(p.get("yards_gained") or 0.0) for p in plays]
            count = len(plays)
            avg = statistics.fmean(yards) if yards else 0.0
            median = statistics.median(yards) if yards else 0.0
            success = sum(1 for p in plays if _success(p))
            success_rate = success / count if count else 0.0
            explosives = sum(1 for p in plays if p.get("explosive"))
            rows.append((metric, val, count, avg, median, success_rate, explosives))
    return rows


# ---------------------------------------------------------------------------
# Output helpers

def write_csv(out_dir: Path, rows: List[Tuple[str, str, int, float, float, float, int]],
              csv_out: str | None = None) -> Path:
    csv_path = Path(csv_out) if csv_out else out_dir / "tendencies.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "count", "avg_yards", "median_yards", "success_rate", "explosives"])
        for r in rows:
            w.writerow(r)
    return csv_path


def write_md(out_dir: Path, rows: List[Tuple[str, str, int, float, float, float, int]]) -> Path:
    total = 0
    rp = Counter()
    forms = Counter()
    dirs = Counter()
    routes = Counter()
    for metric, val, count, *_ in rows:
        if metric == "run_pass":
            rp[val] += count; total += count
        if metric == "formation_text":
            forms[val] += count
        if metric == "run_direction":
            dirs[val] += count
        if metric == "route_primary":
            routes[val] += count

    def fmt(counter: Counter) -> str:
        parts = [f"- {k}: {v}" for k, v in counter.most_common(5)]
        return "\n".join(parts) if parts else "- none"

    lines = ["# Opponent Tendencies", f"**Total plays:** {total}", "", "## Run/Pass", fmt(rp), "", "## Formations", fmt(forms), "", "## Run Direction", fmt(dirs), "", "## Routes", fmt(routes)]
    md_path = out_dir / "tendencies.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# Entry points

def _run(args: argparse.Namespace) -> Tuple[Path, Path]:
    out_dir = Path(args.out_dir)
    plays = load_plays(out_dir)
    plays = filter_plays(
        plays,
        only_off=args.only_lincoln_offense,
        only_def=args.only_lincoln_defense,
        exclude_phase=args.exclude_phase.split(",") if args.exclude_phase else [],
        min_conf=args.min_side_conf,
        use_raw_side=getattr(args, "use_raw_side", False),
    )
    for p in plays:
        gen_form = p.get("generic_formation")
        if not gen_form:
            fr = p.get("frame_meta", {})
            from .generic_formation import infer_generic
            gen_form = infer_generic(fr)
        p["generic_formation"] = gen_form

        yards = p.get("yards")
        if yards is None:
            tr = p.get("carrier_track") or []
            if len(tr) >= 2:
                dy = abs(tr[-1][1] - tr[0][1])
                yards = round(float(dy) * 100.0, 1)
            else:
                yards = 0.0
        p["yards_est"] = yards
        if p.get("yards_gained") is None:
            p["yards_gained"] = yards
    agg = summarise(plays)
    rows = _metric_rows(agg)
    csv_path = write_csv(out_dir, rows, args.csv_out)
    md_path = write_md(out_dir, rows)
    return csv_path, md_path


def run_from_pipeline(out_dir: str, *, only_lincoln_offense: bool = False,
                      only_lincoln_defense: bool = False,
                      exclude_phase: str = "special_teams,unknown",
                      min_side_conf: float = 0.40,
                      csv_out: str | None = None,
                      use_raw_side: bool = False) -> Tuple[Path, Path]:
    args = argparse.Namespace(
        out_dir=out_dir,
        only_lincoln_offense=only_lincoln_offense,
        only_lincoln_defense=only_lincoln_defense,
        exclude_phase=exclude_phase,
        min_side_conf=min_side_conf,
        csv_out=csv_out,
        use_raw_side=use_raw_side,
    )
    return _run(args)


def parse_args() -> argparse.Namespace:  # pragma: no cover - CLI helper
    ap = argparse.ArgumentParser(description="compute opponent tendencies")
    ap.add_argument("out_dir")
    ap.add_argument("--only-lincoln-offense", action="store_true")
    ap.add_argument("--only-lincoln-defense", action="store_true")
    ap.add_argument("--use-raw-side", action="store_true", help="use lincoln_side not *_final")
    ap.add_argument("--exclude-phase", default="special_teams,unknown")
    ap.add_argument("--min-side-conf", type=float, default=0.40)
    ap.add_argument("--csv-out", default=None)
    return ap.parse_args()


def main() -> None:  # pragma: no cover - CLI entry
    _run(parse_args())


if __name__ == "__main__":  # pragma: no cover
    main()

