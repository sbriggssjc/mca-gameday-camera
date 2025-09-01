import json, re
from pathlib import Path

def norm(s): return re.sub(r'\s+', ' ', s.strip())

pb = json.load(open("playbooks/mca_5th_playbook.json"))
plays = []

def walk(x):
    if isinstance(x, dict):
        if "plays" in x and isinstance(x["plays"], list):
            for p in x["plays"]:
                pid = p.get("id") or p.get("name") or p.get("title")
                if isinstance(pid, str): plays.append(norm(pid))
        for v in x.values(): walk(v)
    elif isinstance(x, list):
        for v in x: walk(v)

walk(pb)
plays = sorted(set([p for p in plays if p]))
if not plays:
    raise SystemExit("No plays found in playbook.")

Path("models/play_classifier").mkdir(parents=True, exist_ok=True)
Path("models/formation").mkdir(parents=True, exist_ok=True)  # if you add formation derivation later
Path("models/play_classifier/labels.txt").write_text("\n".join(plays) + "\n")
print(f"wrote models/play_classifier/labels.txt with {len(plays)} labels")
