import csv
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from playbooks import load_offense_playbook

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--playbook", default=None)
    ap.add_argument("--out", default="output/wristband.csv")
    args = ap.parse_args()
    pb = load_offense_playbook(args.playbook)
    slots = pb.get("wristband", {}).get("slots", [])
    legend = pb.get("wristband", {}).get("abbrev_legend", {})
    rows = []
    for s in slots:
        rows.append({
            "Slot": s.get("slot"),
            "Signal": s.get("sig"),
            "Play #": s.get("num"),
            "Name": s.get("name"),
            "Pair": legend.get(s.get("sig"), ""),
        })
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=["Slot", "Signal", "Play #", "Name", "Pair"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
