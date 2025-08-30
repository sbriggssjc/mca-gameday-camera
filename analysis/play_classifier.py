from __future__ import annotations

"""Simple play classification helpers.

These helpers accept either legacy dict playbooks or ``PlaybookIndex`` style
objects.  The classifier itself is intentionally lightweight; it returns
formation information when present and proposes top-N candidate play names
from the supplied playbook.
"""

import math
from typing import Any, Dict, Iterable, List, Tuple
import math

# ---------------------------------------------------------------------------
# Playbook helpers


def _iter_playbook_plays(pb: Any) -> Iterable[Dict[str, Any]]:
    """Yield play dicts from ``pb`` regardless of its structure."""

    if isinstance(pb, dict):
        plays = pb.get("plays", [])
        for p in plays:
            if isinstance(p, dict):
                yield p
        return

    plays = getattr(pb, "plays", None)
    if plays is None and hasattr(pb, "to_dict"):
        plays = pb.to_dict().get("plays", [])
    if plays is None and hasattr(pb, "__dict__"):
        plays = getattr(pb.__dict__, "plays", []) or []

    for p in plays or []:
        if isinstance(p, dict):
            yield p
        else:
            yield {
                "name": getattr(p, "name", ""),
                "formation": getattr(p, "formation", ""),
                "formations": getattr(p, "formations", None),
                "family": getattr(p, "family", ""),
            }


def _tokenize(s: str) -> List[str]:
    return [t for t in s.lower().split() if t]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[lb]


def _name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    tok_a, tok_b = _tokenize(a), _tokenize(b)
    j = _jaccard(tok_a, tok_b)
    lev = _levenshtein(a.lower(), b.lower())
    norm = 1.0 - (lev / max(len(a), len(b), 1))
    return 0.5 * j + 0.5 * max(0.0, norm)


def _best_matches_from_playbook(
    formation: str,
    pb: Any,
    topk: int = 3,
) -> List[Tuple[str, float]]:
    """Return ``topk`` candidate names and scores from ``pb``.

    The scoring function is intentionally very lightweight – it simply
    performs a loose formation match and then scores the candidate name based
    on string similarity.  The resulting "scores" are used downstream as
    pseudo logits for temporal smoothing heuristics.
    """

    cands: List[Tuple[str, float]] = []
    for p in _iter_playbook_plays(pb):
        name = p.get("name") or ""
        if not name:
            continue
        pf = p.get("formation") or ""
        formations = p.get("formations") or []
        if isinstance(formations, str):
            formations = [formations]
        formation_match = False
        if formation and (pf or formations):
            pool = [pf, *formations]
            for f in pool:
                if f and (f.lower() in formation.lower() or formation.lower() in f.lower()):
                    formation_match = True
                    break
        # Compare the formation text with the play name as a loose proxy for a
        # classifier score.  This keeps the implementation deterministic while
        # still yielding a range of confidences for tests to exercise.
        sim = _name_similarity(formation, name)
        score = min(1.0, sim + (0.2 if formation_match else 0.0))
        cands.append((name, score))
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[:topk]


def _best_family_from_playbook(pb: Any, scores: Dict[str, float]) -> str:
    """Return the highest scoring family from ``scores`` using ``pb``."""

    fam_scores: Dict[str, float] = {}
    if not scores:
        return ""
    for p in _iter_playbook_plays(pb):
        name = p.get("name") or ""
        if name in scores:
            family = p.get("family") or ""
            if family:
                fam_scores[family] = fam_scores.get(family, 0.0) + scores[name]
    if not fam_scores:
        return ""
    return max(fam_scores.items(), key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Main classifier


def classify_plays(
    segments: List[Dict[str, float]],
    playbook: Any,
    team: str,
    *,
    play_ckpt: str | None = None,
    play_labels: str | None = None,
    formation_ckpt: str | None = None,
    formation_labels: str | None = None,
    weak_threshold: float = 0.35,
    smooth_frames: int = 4,
) -> List[Dict[str, Any]]:
    """Classify each segment and propose candidate play names.

    In addition to the top candidate, this helper also records a list of top-3
    candidates, detects low-confidence ("weak") predictions, applies a simple
    temporal smoothing fallback and, if necessary, backs off to family-level
    classification.
    """

    # The ``*_ckpt`` and ``*_labels`` arguments are accepted for API
    # compatibility with the full model-backed implementation.  They are not
    # used in this lightweight fallback classifier.
    _ = (play_ckpt, play_labels, formation_ckpt, formation_labels)

    # ------------------------------------------------------------------
    # Pre-compute candidate score dictionaries for all segments so that
    # temporal smoothing can operate on them.
    # ------------------------------------------------------------------
    raw_scores: List[Dict[str, float]] = []
    formations: List[str] = []
    for seg in segments:
        formation = seg.get("formation", "") or ""
        formations.append(formation)
        # Always propose candidates irrespective of formation or activity
        cands = _best_matches_from_playbook(formation, playbook, topk=5)
        if not cands:
            cands = _best_matches_from_playbook("", playbook, topk=5)
        raw_scores.append({n: s for n, s in cands})

    def _softmax(d: Dict[str, float]) -> Dict[str, float]:
        if not d:
            return {}
        m = max(d.values())
        exps = {k: math.exp(v - m) for k, v in d.items()}
        total = sum(exps.values()) or 1.0
        return {k: v / total for k, v in exps.items()}

    results: List[Dict[str, Any]] = []
    total = len(segments)
    for i, seg in enumerate(segments, 1):
        logits = raw_scores[i - 1]
        probs = _softmax(logits)
        sorted_cands = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_name, top_score = (sorted_cands[0] if sorted_cands else ("", 0.0))
        final_probs = probs
        smoothing_applied = 0

        if smooth_frames > 0:
            start = max(0, (i - 1) - smooth_frames)
            end = min(total, (i - 1) + smooth_frames + 1)
            window = raw_scores[start:end]
            names = {n for d in window for n in d}
            avg_logits: Dict[str, float] = {}
            for n in names:

                smoothed[n] = sum(d.get(n, 0.0) for d in window) / len(window)
            if smoothed:
                sorted_cands = sorted(smoothed.items(), key=lambda x: x[1], reverse=True)
                final_scores = smoothed
                top_name, top_score = (sorted_cands[0] if sorted_cands else ("", 0.0))

        # Convert scores to probabilities
        if final_scores:
            max_logit = max(final_scores.values())
            exp_scores = {k: math.exp(v - max_logit) for k, v in final_scores.items()}
            total = sum(exp_scores.values()) or 1.0
            probs = {k: v / total for k, v in exp_scores.items()}
        else:
            probs = {}
        sorted_cands = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_name, top_score = (sorted_cands[0] if sorted_cands else ("", 0.0))

                avg_logits[n] = sum(d.get(n, 0.0) for d in window) / len(window)
            smoothed_probs = _softmax(avg_logits)
            smoothed_sorted = sorted(smoothed_probs.items(), key=lambda x: x[1], reverse=True)
            if smoothed_sorted and smoothed_sorted[0][1] > top_score:
                final_probs = smoothed_probs
                sorted_cands = smoothed_sorted
                top_name, top_score = smoothed_sorted[0]
                smoothing_applied = 1


        if top_score < weak_threshold:
            clf_family = _best_family_from_playbook(playbook, final_probs)
        else:
            clf_family = ""

        weak_flag = 1 if top_score < weak_threshold else 0

        top5 = sorted_cands[:5]
        top3 = sorted_cands[:3]

        results.append(
            {
                "play_id": seg.get("id") or seg.get("play_id") or f"PLAY_{i:03d}",
                "formation": formations[i - 1],
                "formation_confidence": float(seg.get("formation_confidence", 0.0)),
                # Existing fields for backwards compatibility
                "play_family": top_name,
                "playcall_confidence": float(top_score),
                "candidates": top5,
                "outcome": seg.get("outcome", ""),
                # New observability fields
                "clf_top1": top_name,
                "clf_top1_conf": float(top_score),
                "clf_top3": top3,
                "clf_weak_flag": weak_flag,
                "clf_family": clf_family,
                "smoothing_applied": smoothing_applied,
            }
        )
    return results


__all__ = ["classify_plays"]
