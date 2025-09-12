from __future__ import annotations
import argparse, json, pathlib

HELP = """Examples:
# by indices (1-based):
python -m analysis.seed_labels OUT --offense 7 9 15 --defense 3 4 --special 20 21

# by filenames (each file is its own argument after --files):
python -m analysis.seed_labels OUT --files --offense "Wide - Clip 007.mp4" "Wide - Clip 009.mp4"
"""


def load_rows(out: pathlib.Path):
    p = out / "plays.jsonl"
    if not p.exists():
        raise SystemExit(
            f"[seed] plays.jsonl not found at '{p}'. Set OUT to your pipeline run dir."
        )
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def build_src_index(rows):
    srcs = [r["src"] for r in rows if isinstance(r, dict) and "src" in r]
    by_name = {pathlib.Path(s).name: s for s in srcs}
    return srcs, by_name


def main():
    ap = argparse.ArgumentParser(
        description="Seed labels for side-of-ball",
        epilog=HELP,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("out_dir")
    ap.add_argument(
        "--files",
        action="store_true",
        help="Treat following items as filenames instead of indices",
    )
    ap.add_argument("--offense", nargs="*", default=[], help="Indices or filenames for offense")
    ap.add_argument("--defense", nargs="*", default=[], help="Indices or filenames for defense")
    ap.add_argument("--special", nargs="*", default=[], help="Indices or filenames for special teams")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    rows = load_rows(out)
    srcs, by_name = build_src_index(rows)

    def to_src_list(items, label):
        out_list = []
        for it in items:
            if args.files:
                s = by_name.get(it)
                if not s:
                    print(f"[seed] WARN: filename not found in this run: {it}")
                    continue
                out_list.append(s)
            else:
                try:
                    j = int(it) - 1
                    if 0 <= j < len(srcs):
                        out_list.append(srcs[j])
                    else:
                        print(f"[seed] WARN: index out of range for {label}: {it}")
                except ValueError:
                    print(f"[seed] WARN: not an int index (missing --files?): {it}")
        return out_list

    seeds: dict[str, str] = {}
    for s in to_src_list(args.offense, "offense"):
        seeds[s] = "offense"
    for s in to_src_list(args.defense, "defense"):
        seeds[s] = "defense"
    for s in to_src_list(args.special, "special"):
        seeds[s] = "special_teams"

    seed_path = out / "seed_labels.json"
    if seed_path.exists():
        existing = json.loads(seed_path.read_text())
        existing.update(seeds)
        seeds = existing

    seed_path.write_text(json.dumps(seeds, indent=2))
    print("[seed] merged and wrote", seed_path)


if __name__ == "__main__":
    main()

