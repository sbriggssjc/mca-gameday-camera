from __future__ import annotations

"""Hidden Markov Model smoothing of offense/defense classifications."""

import json
import pathlib
import sys
from typing import Dict, List


STATES = ["offense", "defense"]
# Drives rarely flip back and forth quickly; favour staying in the same state
LOG_P_STAY = -0.1
LOG_P_SWITCH = -2.0


def _viterbi(obs: List[Dict[str, float]]) -> List[str]:
    """Classic Viterbi decoding for a two-state HMM."""

    V: List[Dict[str, tuple[float, str | None]]] = [
        {s: (-1e9, None) for s in STATES} for _ in obs
    ]

    for s in STATES:
        V[0][s] = (obs[0][s], None)

    for t in range(1, len(obs)):
        for s in STATES:
            best = (-1e9, None)
            for sp in STATES:
                trans = LOG_P_STAY if s == sp else LOG_P_SWITCH
                score = V[t - 1][sp][0] + trans + obs[t][s]
                if score > best[0]:
                    best = (score, sp)
            V[t][s] = best

    last_state = max(STATES, key=lambda s: V[-1][s][0])
    path = [last_state]
    for t in range(len(obs) - 1, 0, -1):
        last_state = V[t][last_state][1]  # type: ignore[index]
        path.append(last_state)  # type: ignore[arg-type]
    return list(reversed(path))


def smooth(out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    p = out / "plays.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

    idx_map: List[int] = []
    obs: List[Dict[str, float]] = []
    for i, r in enumerate(rows):
        if r.get("phase") == "special_teams":
            continue
        side = r.get("lincoln_side", "unknown")
        c = float(r.get("lincoln_side_conf", 0.3))
        ll = {"offense": -1.2, "defense": -1.2}
        if side == "offense":
            ll["offense"] = -0.1 + 1.5 * c
        elif side == "defense":
            ll["defense"] = -0.1 + 1.5 * c
        obs.append(ll)
        idx_map.append(i)

    if not obs:
        return

    path = _viterbi(obs)
    for state, idx in zip(path, idx_map):
        rows[idx]["lincoln_side_smoothed"] = state

    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[sequence_smooth] wrote smoothed sides")


if __name__ == "__main__":  # pragma: no cover
    smooth(sys.argv[1] if len(sys.argv) > 1 else "output")

