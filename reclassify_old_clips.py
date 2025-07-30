# Reclassify previously labeled highlight clips using the latest model
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
    from torchvision import transforms
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

from play_inference import ToFloatNormalize, load_model, read_clip

LOG_PATH = Path("logs/learning_log.json")


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load clip metadata from CSV."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_dataset(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write updated metadata back to the CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_latest_model(dir_path: Path) -> Path:
    """Return newest ``.pt`` file in ``dir_path``."""
    ckpts = sorted(dir_path.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No model checkpoint found in {dir_path}")
    return ckpts[0]


def append_log(entry: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
    except Exception:
        data = []
    data.append(entry)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def reclassify(
    dataset_path: Path,
    model_dir: Path = Path("models/play_classifier"),
    clip_len: int = 16,
    dry_run: bool = False,
) -> None:
    """Reclassify clips and update labels if predictions change."""
    if torch is None:
        raise ImportError("PyTorch is required for reclassification")

    entries = load_dataset(dataset_path)
    if not entries:
        print("[!] No clips found in dataset")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = find_latest_model(model_dir)
    model, inv_map = load_model(str(model_path), device)
    transform = transforms.Compose([transforms.Resize((224, 224)), ToFloatNormalize()])

    dataset_dir = dataset_path.parent
    updated = False
    for item in entries:
        rel_clip = Path(item["filepath"])
        clip_path = rel_clip if rel_clip.is_absolute() else dataset_dir / rel_clip
        old_label = item.get("label", "")
        player = item.get("player", "")
        quarter = item.get("quarter", "")
        time_code = item.get("time", "")

        clip = read_clip(clip_path, clip_len, transform)
        clip = clip.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(clip)
        pred_idx = output.argmax(1).item()
        new_label = inv_map.get(pred_idx, "unknown")
        if new_label != old_label:
            print(f"Updated: {item['filepath']} | {old_label} → {new_label}")
            updated = True
            if not dry_run:
                item["label"] = new_label
                new_name = f"{new_label.replace(' ', '_')}_{player.replace(' ', '')}_Q{quarter}_{time_code}{clip_path.suffix}"
                new_path = clip_path.with_name(new_name)
                new_path.parent.mkdir(parents=True, exist_ok=True)
                clip_path.rename(new_path)
                # store path relative to dataset directory
                try:
                    item["filepath"] = str(new_path.relative_to(dataset_dir))
                except ValueError:
                    item["filepath"] = str(new_path)

                log_entry = {
                    "clip": clip_path.name,
                    "old_label": old_label,
                    "new_label": new_label,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                append_log(log_entry)

    if updated and not dry_run:
        save_dataset(dataset_path, entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reclassify highlight clips")
    parser.add_argument(
        "dataset",
        default="data/highlights/labels.csv",
        nargs="?",
        help="Path to CSV with clip metadata",
    )
    parser.add_argument(
        "--model_dir",
        default="models/play_classifier",
        help="Directory with model checkpoints",
    )
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    args = parser.parse_args()

    reclassify(Path(args.dataset), Path(args.model_dir), args.clip_len, args.dry_run)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
