#!/usr/bin/env python3
"""Normalize Jenks/Metro side of ball based on the audited CSV template."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Row = Dict[str, Any]
Play = Dict[str, Any]

ROW_KEY_PREFERENCES: Sequence[str] = (
    "play_id",
    "idx",
    "index",
    "play_index",
    "row_id",
    "play_number",
)

OFFENSE_RE = re.compile(r"\boff(?:en[cs]e)?\b|\boff_?team\b", re.IGNORECASE)
DEFENSE_RE = re.compile(r"\bdef(?:en[cs]e)?\b|\bdef_?team\b", re.IGNORECASE)
SIDE_HINT_RE = re.compile(r"(side|offen|defen)", re.IGNORECASE)
TEAM_RE = re.compile(r"(team|opponent|opp|school|home|away|vs|versus|against)", re.IGNORECASE)
JENKS_RE = re.compile(r"jenks", re.IGNORECASE)

SIDE_VALUE_MAP = {
    "o": "offense",
    "off": "offense",
    "offense": "offense",
    "offence": "offense",
    "offensive": "offense",
    "d": "defense",
    "def": "defense",
    "defense": "defense",
    "defence": "defense",
    "defensive": "defense",
}

CLIP_FIELDS = ("clip", "src", "name", "title", "file")


def norm_side(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    if text == "unknown":
        return "unknown"
    return SIDE_VALUE_MAP.get(text, "")


def is_special(row: Row) -> bool:
    # Treat any non-empty st_* as special teams
    return bool(str(row.get("st_fix") or row.get("st_auto") or "").strip())


def is_excluded(row: Row) -> bool:
    ex = str(row.get("exclude") or "").strip().lower()
    return ex in {"1", "true", "yes", "y"}


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _preferred_column(fieldnames: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    normalized = { _norm_name(field): field for field in fieldnames if field }
    for cand in candidates:
        key = _norm_name(cand)
        if key in normalized:
            return normalized[key]
    return None


def _collect_columns(fieldnames: Iterable[str], pattern: re.Pattern[str]) -> List[str]:
    cols: List[str] = []
    for name in fieldnames:
        if name and pattern.search(name):
            cols.append(name)
    return cols


def _clip_number(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value)
    match = re.search(r"clip\s*[-_ ]?\s*(\d{1,4})", text, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    digits = re.findall(r"\d{1,4}", text)
    if digits:
        try:
            return int(digits[-1])
        except ValueError:
            return None
    return None


def _row_mentions_jenks(row: Row, candidate_cols: Sequence[str]) -> bool:
    if candidate_cols:
        values = [row.get(col, "") for col in candidate_cols]
    else:
        values = list(row.values())
    for val in values:
        if JENKS_RE.search(str(val or "")):
            return True
    return False


def _side_from_tokens(tokens: Iterable[str]) -> Optional[str]:
    for tok in tokens:
        key = SIDE_VALUE_MAP.get(tok)
        if key:
            return key
    return None


def _tokenize(value: Any) -> List[str]:
    if value is None:
        return []
    text = str(value).strip().lower()
    if not text:
        return []
    tokens = [t for t in re.split(r"[^a-z]+", text) if t]
    # include exact string for single-letter cases like "O" or "D"
    if len(text) == 1 and text in {"o", "d"}:
        tokens.append(text)
    if text in {"off", "def"}:
        tokens.append(text)
    return tokens


def determine_row_side(row: Row, offense_cols: Sequence[str], defense_cols: Sequence[str],
                        hint_cols: Sequence[str], jenks_cols: Sequence[str]) -> str:
    offense_hit = any(JENKS_RE.search(str(row.get(col, ""))) for col in offense_cols)
    defense_hit = any(JENKS_RE.search(str(row.get(col, ""))) for col in defense_cols)
    if offense_hit and defense_hit:
        return "conflict"
    if offense_hit:
        return "offense"
    if defense_hit:
        return "defense"

    if not _row_mentions_jenks(row, jenks_cols):
        return "unknown"

    for col in hint_cols:
        tokens = _tokenize(row.get(col))
        side = _side_from_tokens(tokens)
        if side:
            return side
    return "unknown"


def load_audit_rows(audit_path: Path) -> Tuple[List[Row], List[str]]:
    if not audit_path.exists():
        raise SystemExit(f"[error] audit file not found: {audit_path}")
    with audit_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows: List[Row] = list(reader)
    return rows, fieldnames


def load_plays(plays_path: Path) -> List[Tuple[str, Optional[Play]]]:
    entries: List[Tuple[str, Optional[Play]]] = []
    if not plays_path.exists():
        raise SystemExit(f"[error] plays file not found: {plays_path}")
    with plays_path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                entries.append((raw, None))
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                entries.append((raw, None))
                continue
            entries.append((raw, obj))
    return entries


def build_row_index(
    rows: List[Row], fieldnames: List[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str], Dict[int, str], Dict[str, str]]:
    key_col = _preferred_column(fieldnames, ROW_KEY_PREFERENCES)
    if key_col is None:
        key_col = "row_index"
        for idx, row in enumerate(rows):
            row[key_col] = str(idx)
    offense_cols = _collect_columns(fieldnames, OFFENSE_RE)
    defense_cols = _collect_columns(fieldnames, DEFENSE_RE)
    hint_cols = _collect_columns(fieldnames, SIDE_HINT_RE)

    jenks_cols = list(dict.fromkeys(_collect_columns(fieldnames, TEAM_RE) + offense_cols + defense_cols))
    if not jenks_cols:
        for fallback in ("clip", "src", "title"):
            col = _preferred_column(fieldnames, (fallback,))
            if col:
                jenks_cols.append(col)

    side_by_key: Dict[str, str] = {}
    side_by_row: Dict[str, str] = {}
    side_by_clip: Dict[int, str] = {}
    side_map: Dict[str, str] = {}
    ordered: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        detected_side = determine_row_side(row, offense_cols, defense_cols, hint_cols, jenks_cols)
        rid = str(row.get(key_col, "")).strip()
        if not rid:
            rid = str(idx)
        clip_num = None
        for field in CLIP_FIELDS:
            clip_num = _clip_number(row.get(field))
            if clip_num is not None:
                break
        if is_special(row):
            side = "special"
        elif is_excluded(row):
            side = "excluded"
        else:
            fixed_side = norm_side(row.get("side_fix")) or norm_side(row.get("side_auto"))
            if fixed_side and fixed_side != "unknown":
                side = fixed_side
            else:
                side = detected_side
            if fixed_side == "unknown" and detected_side in {"offense", "defense"}:
                side = detected_side
        if not side or side == "conflict":
            side = "unknown"
        info = {
            "key": rid,
            "row_index": str(idx),
            "side": side,
            "clip": clip_num,
        }
        ordered.append(info)
        side_map[rid] = side
        side_map[str(idx)] = side
        if clip_num is not None:
            side_map[f"clip:{clip_num}"] = side
        side_by_key[rid] = side
        side_by_row[str(idx)] = side
        if clip_num is not None:
            side_by_clip[clip_num] = side
    return ordered, side_by_key, side_by_row, side_by_clip, side_map


def match_side(
    play: Play,
    idx: int,
    key_map: Dict[str, str],
    row_map: Dict[str, str],
    clip_map: Dict[int, str],
    candidate_fields: Sequence[str],
) -> Tuple[Optional[str], Optional[str]]:
    for field in candidate_fields:
        if field not in play:
            continue
        val = play.get(field)
        if val is None:
            continue
        key = str(val).strip()
        if key and key in key_map:
            return key, key_map[key]
    clip_num = None
    for field in CLIP_FIELDS:
        clip_num = _clip_number(play.get(field))
        if clip_num is not None:
            break
    if clip_num is not None and clip_num in clip_map:
        return f"clip:{clip_num}", clip_map[clip_num]
    row_key = str(idx)
    if row_key in row_map:
        return row_key, row_map[row_key]
    return None, None


def write_plays(plays_path: Path, entries: List[Tuple[str, Optional[Play]]]) -> None:
    text = "\n".join(
        json.dumps(obj, ensure_ascii=False) if obj is not None else raw
        for raw, obj in entries
    )
    plays_path.write_text(text + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Set jenks_side/metro_side based on audit CSV")
    default_out = os.environ.get("OUT")
    parser.add_argument("out", nargs="?", default=default_out, help="Output directory containing plays.jsonl")
    parser.add_argument("--audit", help="Path to audit_template.csv", default=None)
    args = parser.parse_args(argv)

    if not args.out:
        parser.error("OUT directory not provided")
    out_dir = Path(args.out)
    plays_path = out_dir / "plays.jsonl"
    audit_path = Path(args.audit) if args.audit else out_dir / "audit" / "audit_template.csv"

    rows, fieldnames = load_audit_rows(audit_path)
    if not rows:
        print("[warn] audit CSV empty; no side updates applied")
        return
    _, side_by_key, side_by_row, side_by_clip, side_map = build_row_index(rows, fieldnames)

    entries = load_plays(plays_path)
    candidate_fields: List[str] = []
    seen_fields = set()
    for pref in ROW_KEY_PREFERENCES:
        norm_pref = _norm_name(pref)
        for _, obj in entries:
            if obj is None:
                continue
            for field in obj.keys():
                if _norm_name(str(field)) == norm_pref:
                    if field not in seen_fields:
                        candidate_fields.append(field)
                        seen_fields.add(field)
        if seen_fields:
            break
    if not candidate_fields:
        candidate_fields = ["play_id", "index", "idx", "segment_id"]

    backup = plays_path.parent / f"{plays_path.name}.bak"
    shutil.copyfile(plays_path, backup)

    summary = Counter()
    updated = 0
    for idx, (raw, obj) in enumerate(entries):
        if obj is None:
            continue
        play_key, _ = match_side(
            obj, idx, side_by_key, side_by_row, side_by_clip, candidate_fields
        )
        js = side_map.get(play_key, "unknown")
        if not js:
            js = "unknown"
        if js in {"offense", "defense"}:
            obj["jenks_side"] = js
            obj["metro_side"] = "defense" if js == "offense" else "offense"
            updated += 1
        else:
            obj["jenks_side"] = js
            obj.pop("metro_side", None)
        summary[js] += 1
        entries[idx] = (raw, obj)

    write_plays(plays_path, entries)
    counts = {k: summary[k] for k in sorted(summary)}
    print(f"Jenks side counts: {counts}")
    print(f"Updated plays: {updated}")


if __name__ == "__main__":  # pragma: no cover
    main()
