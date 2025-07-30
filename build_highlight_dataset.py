import argparse
import csv
import re
import shutil
from pathlib import Path


def parse_filename(filename: str) -> dict | None:
    """Parse highlight filename into metadata.

    Expected format is ``PREFIX_PLAYER_PLAYTYPE_Q<quarter>_<time>.mp4`` where
    ``time`` is ``05m12s``. Additional underscores in the play type are
    converted to spaces.
    """
    name = Path(filename).stem
    parts = name.split("_")
    if len(parts) < 5:
        return None
    # find the quarter token (e.g. "Q2")
    q_index = next((i for i, p in enumerate(parts) if p.startswith("Q")), -1)
    if q_index == -1 or q_index + 1 >= len(parts):
        return None
    player = parts[1]
    label = " ".join(parts[2:q_index]).replace("-", " ")
    quarter = parts[q_index][1:]
    time_match = re.match(r"(?P<m>\d+)m(?P<s>\d+)s", parts[q_index + 1])
    if not time_match:
        return None
    time = f"{int(time_match.group('m')):02d}:{int(time_match.group('s')):02d}"
    return {"player": player, "label": label, "quarter": f"Q{quarter}", "time": time}


def build_dataset(src_dir: Path, dest_dir: Path) -> Path:
    """Copy clips from ``src_dir`` into ``dest_dir`` and write metadata CSV."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "quarter", "time", "player"])
        for mp4 in sorted(src_dir.glob("*.mp4")):
            info = parse_filename(mp4.name)
            if not info:
                print(f"[!] Skipping {mp4.name}")
                continue
            dest = dest_dir / mp4.name
            if mp4.resolve() != dest.resolve():
                shutil.copy2(mp4, dest)
            writer.writerow([str(dest), info["label"], info["quarter"], info["time"], info["player"]])
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build highlight training dataset")
    parser.add_argument("source", help="folder containing highlight clips")
    parser.add_argument("dest", help="destination dataset folder")
    args = parser.parse_args()
    src = Path(args.source)
    dest = Path(args.dest)
    csv_path = build_dataset(src, dest)
    print(f"\u2705 Dataset written to {csv_path}")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
