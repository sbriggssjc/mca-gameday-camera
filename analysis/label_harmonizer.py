from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Tuple

# Paths to playbook and optional alias mappings relative to repo root
BASE_DIR = Path(__file__).resolve().parent.parent
PLAYBOOK_PATH = BASE_DIR / "playbooks" / "mca_5th_playbook.json"
ALIAS_PATH = BASE_DIR / "playbooks" / "label_aliases.json"

STOPWORDS = {"play", "run", "pass"}


def normalize(s: str) -> str:
    """Return normalized label string.

    Steps: lowercase, strip, replace '_'/'-' with space, collapse whitespace,
    drop punctuation (anything not ``a-z0-9 ``) and remove trailing stopwords.
    """
    if not s:
        return ""
    s = s.lower().strip().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    while tokens and tokens[-1] in STOPWORDS:
        tokens.pop()
    return " ".join(tokens)


def _load_aliases(path: Path = ALIAS_PATH) -> dict[str, str]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _build_playbook_index(path: Path = PLAYBOOK_PATH) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    try:
        pb = json.loads(path.read_text())
        for play in pb.get("plays", []):
            name = play.get("name")
            if name:
                index[name] = set(normalize(name).split())
    except Exception:
        pass
    return index


ALIASES = _load_aliases()
PLAYBOOK_INDEX = _build_playbook_index()
CANON_NORMALIZED = {normalize(name): name for name in PLAYBOOK_INDEX}


def harmonize(
    model_label: str,
    playbook_index: dict[str, set[str]] | None = None,
    aliases: dict[str, str] | None = None,
) -> Tuple[str | None, str]:
    """Map ``model_label`` to a canonical playbook name and reason."""
    if aliases is None:
        aliases = ALIASES
    if playbook_index is None:
        playbook_index = PLAYBOOK_INDEX
    if not model_label:
        return None, "unmapped"
    if model_label in aliases:
        return aliases[model_label], "alias"
    norm = normalize(model_label)
    if norm in CANON_NORMALIZED:
        return CANON_NORMALIZED[norm], "exact"
    tokens = set(norm.split()) if norm else set()
    if not tokens:
        return None, "unmapped"
    best = None
    best_score = 0.0
    for canon, canon_tokens in playbook_index.items():
        if not canon_tokens:
            continue
        overlap = tokens & canon_tokens
        score = len(overlap) / len(tokens)
        if score >= 0.8 and score > best_score:
            best = canon
            best_score = score
    if best:
        return best, "token_match"
    return None, "unmapped"


def map_topk(labels_with_scores: Iterable[Tuple[str, float]]):
    """Return canonical mapping for top1 and top3 labels.

    ``labels_with_scores`` is an iterable of ``(label, score)`` pairs ordered by
    score.  The scores are ignored; only labels are used.
    """
    canon_labels: list[str] = []
    reason = "unmapped"
    for i, (label, _score) in enumerate(labels_with_scores):
        canon, r = harmonize(label)
        canon_labels.append(canon or "")
        if i == 0:
            reason = r
    canon_top1 = canon_labels[0] if canon_labels else ""
    canon_top3_string = "|".join(canon_labels)
    return canon_top1, canon_top3_string, reason
