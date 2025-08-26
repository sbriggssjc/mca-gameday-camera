from __future__ import annotations
import json, sys, pathlib
def main(path):
    p = pathlib.Path(path)
    report = json.loads(p.read_text())
    for play in report.get("plays", []):
        play.setdefault("playcall_candidates", play.get("candidates", []))
    p.write_text(json.dumps(report, indent=2))
if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

