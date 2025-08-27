from __future__ import annotations

import json, sys, pathlib


def main(path: str) -> None:
    p = pathlib.Path(path)
    report = json.loads(p.read_text())
    for play in report.get("plays", []):
        cands = []
        pc = play.get("playcall") or {}
        if isinstance(pc, dict):
            cands = pc.get("candidates") or []
        play.setdefault("playcall_candidates", cands)
    p.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

