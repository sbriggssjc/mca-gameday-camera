import json, csv, argparse, pathlib
ap = argparse.ArgumentParser()
ap.add_argument("--formations-csv", required=True)
ap.add_argument("--plays-csv", required=True)
ap.add_argument("--out", default="mca_full_playbook_final.json")
args = ap.parse_args()

def load_csv(path): return list(csv.DictReader(open(path,newline="",encoding="utf8")))
forms = load_csv(args.formations_csv)
plays = load_csv(args.plays_csv)

pb = {"formations": [], "plays": []}
for f in forms:
    pb["formations"].append({
        "name": f["name"],
        "personnel": f.get("personnel") or None,
        "side": f.get("side") or None,
        "tags": [t.strip() for t in (f.get("tags","" ).split("|") ) if t.strip()],
        "anchors": {"vec": [float(x) for x in f.get("vec","" ).split(",") if x]},
    })
for p in plays:
    pb["plays"].append({
        "name": p["name"],
        "formation": p["formation"],
        "family": p.get("family") or None,
        "motion": p.get("motion") or None,
        "tags": [t.strip() for t in (p.get("tags","" ).split("|") ) if t.strip()],
        "cues": { "flow": p.get("flow") or None, "attack_gap": p.get("attack_gap") or None },
    })

pathlib.Path(args.out).write_text(json.dumps(pb, indent=2))
print("Wrote", args.out)
