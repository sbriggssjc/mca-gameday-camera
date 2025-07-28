import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

JERSEY_PATH = Path("./training/labels/confirmed_jerseys.json")
PLAY_PATH = Path("./training/labels/confirmed_play_types.json")
CLEANED_JERSEY_PATH = Path("./training/labels/cleaned_jerseys.json")
CLEANED_PLAY_PATH = Path("./training/labels/cleaned_play_types.json")
LOG_PATH = Path("./training/logs/label_cleaning_log.json")


def load_entries(path: Path) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def clean_entries(entries: List[Dict], label_key: str) -> Tuple[List[Dict], int, int, int]:
    cleaned: List[Dict] = []
    duplicates_removed = 0
    conflicts_found = 0
    empty_labels_removed = 0
    seen: Dict[str, str] = {}

    for item in entries:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        label = item.get(label_key)
        if filename is None:
            continue
        if label is None or label == "":
            empty_labels_removed += 1
            continue
        if filename in seen:
            if seen[filename] == label:
                duplicates_removed += 1
                continue
            conflicts_found += 1
            continue
        seen[filename] = label
        cleaned.append(item)
    return cleaned, duplicates_removed, conflicts_found, empty_labels_removed


def write_json(path: Path, data: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main(apply: bool = False) -> None:
    jerseys = load_entries(JERSEY_PATH)
    plays = load_entries(PLAY_PATH)

    cleaned_jerseys, dup_j, conf_j, empty_j = clean_entries(jerseys, "jersey_number")
    cleaned_plays, dup_p, conf_p, empty_p = clean_entries(plays, "play_type")

    write_json(CLEANED_JERSEY_PATH, cleaned_jerseys)
    write_json(CLEANED_PLAY_PATH, cleaned_plays)

    log_data = {
        "duplicates_removed": dup_j + dup_p,
        "conflicts_found": conf_j + conf_p,
        "empty_labels_removed": empty_j + empty_p,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    if apply:
        if CLEANED_JERSEY_PATH.is_file():
            shutil.move(str(CLEANED_JERSEY_PATH), str(JERSEY_PATH))
        if CLEANED_PLAY_PATH.is_file():
            shutil.move(str(CLEANED_PLAY_PATH), str(PLAY_PATH))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and validate label files")
    parser.add_argument(
        "--apply", action="store_true", help="Overwrite original files with cleaned versions"
    )
    args = parser.parse_args()
    main(apply=args.apply)
