from __future__ import annotations

"""Lightweight play classifier with caching and heuristics."""

from pathlib import Path
from typing import Dict, Any


class PlayClassifier:
    def __init__(self, playbook_cfg, device="cpu", cache_dir=".cache"):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.alias = {
            "Reo F Stick": "Leo F Stick",  # family canonicalization (both -> F Stick)
            "F Stick": "F Stick",
            "Flare Boot": "Flare Boot",
            "Flood": "Flood",
            "Quick Screen": "Quick Screen",
            "F Screen": "F Screen",
            "Jet Sweep": "Jet Sweep",
            "Dive": "Dive",
            "Power R": "Power R",
            "8 Option": "8 Option",
            "Counter": "Counter",
        }
        self._load_index(playbook_cfg)

    def _load_index(self, playbook_cfg):
        # Build or load a simple embedding index for all plays in the playbook
        import json, hashlib, pickle

        pb_json = json.dumps(playbook_cfg, sort_keys=True).encode("utf-8")
        key = hashlib.md5(pb_json).hexdigest()
        cache_path = self.cache_dir / f"play_index_{key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                self.index = pickle.load(f)
            return
        # Build fresh
        plays = playbook_cfg.get("plays", [])

        # Minimal embedding = bag-of-words / keywords for robustness (no external models assumed)
        def featurize(p):
            # concat formation + name + tags
            txt = " ".join([
                str(p.get("formation", "")),
                str(p.get("name", "")),
                " ".join(p.get("tags", [])),
            ]).lower()
            return set(txt.split())

        self.index = [
            {
                "name": p.get("name", ""),
                "family": p.get("family", ""),
                "formation": p.get("formation", ""),
                "feat": featurize(p),
            }
            for p in plays
        ]
        with open(cache_path, "wb") as f:
            pickle.dump(self.index, f)

    def classify(self, segment: Dict[str, Any], formation_hint: str | None = None):
        """Return playcall classification with heuristics.

        Returns a dict:
          {"name": <str>, "confidence": <float>, "family": <str>, "candidates": [ ... ]}
        Never gate on jersey numbers. Use formation and motion heuristics if needed.
        """

        text_bits = []
        if formation_hint:
            text_bits.append(formation_hint.lower())
        if segment.get("motion"):
            text_bits.append(str(segment["motion"]).lower())
        if segment.get("notes"):
            text_bits.append(str(segment["notes"]).lower())
        feat = set(" ".join(text_bits).split())

        # simple score = Jaccard overlap against playbook features
        scored = []
        for row in (self.index or []):
            inter = len(feat & row["feat"])
            union = max(1, len(feat | row["feat"]))
            score = inter / union
            if formation_hint and row["formation"] and formation_hint.lower() in row["formation"].lower():
                score += 0.15
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Fallback heuristics: unique family by formation = high confidence
        if formation_hint:
            fh = formation_hint.lower()
            if "trips" in fh:
                if "jet" in feat or "orbit" in feat:
                    choice = ("Jet Sweep", 0.82)
                elif "flare" in feat or "boot" in feat:
                    choice = ("Flare Boot", 0.84)
                elif "screen" in feat:
                    choice = ("Quick Screen", 0.82)
                else:
                    choice = ("F Stick", 0.90)
                fam = choice[0]
                return {
                    "name": f"Leo {fam}" if "left" in fh else f"Reo {fam}",
                    "confidence": choice[1],
                    "family": fam,
                    "candidates": [],
                }

        if scored and scored[0][0] >= 0.10:
            best = scored[0][1]
            fam = self.alias.get(best["family"] or best["name"], best["family"] or "Unknown")
            conf = min(0.98, max(0.70, 0.70 + scored[0][0]))
            return {
                "name": best["name"] or fam,
                "confidence": conf,
                "family": fam,
                "candidates": [r[1]["name"] for r in scored[:5]],
            }

        return {"name": "Unknown", "confidence": 0.0, "family": "", "candidates": []}


__all__ = ["PlayClassifier"]

