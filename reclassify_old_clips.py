# Reclassify previously labeled highlight clips using the latest model
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
    from torchvision import transforms
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore

from play_inference import ToFloatNormalize, load_model, read_clip

LOG_PATH = Path("training/logs/learning_log.json")


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load clip metadata from CSV or JSON."""
    if path.suffix.lower() == ".csv":
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    with open(path) as f:
        return json.load(f)


def save_dataset(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write updated metadata."""
    if path.suffix.lower() == ".csv":
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)


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
    rename: bool = False,
    regenerate_summary: bool = False,
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

    updated = False
    for item in entries:
        clip_path = Path(item["filepath"])
        old_label = item.get("label", "")
        clip = read_clip(clip_path, clip_len, transform)
        clip = clip.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(clip)
        pred_idx = output.argmax(1).item()
        new_label = inv_map.get(pred_idx, "unknown")
        if new_label != old_label:
            print(f"[update] {clip_path.name}: {old_label} -> {new_label}")
            item["label"] = new_label
            log_entry = {
                "clip": str(clip_path),
                "old_label": old_label,
                "new_label": new_label,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            append_log(log_entry)
            updated = True
            if rename:
                new_path = clip_path.with_name(f"{clip_path.stem}_{new_label}{clip_path.suffix}")
                clip_path.rename(new_path)
                item["filepath"] = str(new_path)

    if updated:
        save_dataset(dataset_path, entries)
        if regenerate_summary:
            try:
                from generate_coaches_cut_and_summary import generate_summary

                generate_summary(Path("analysis"))
            except Exception:
                print("[!] Failed to regenerate summary report")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reclassify highlight clips")
    parser.add_argument("dataset", help="Path to CSV or JSON with clip metadata")
    parser.add_argument(
        "--model_dir", default="models/play_classifier", help="Directory with model checkpoints"
    )
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--rename", action="store_true", help="Rename clip when label changes")
    parser.add_argument(
        "--summary", action="store_true", help="Regenerate summary report if any label changes"
    )
    parser.add_argument(
        "--schedule", action="store_true", help="Run weekly instead of once"
    )
    args = parser.parse_args()

    if args.schedule:
        try:
            import schedule
        except Exception:
            raise ImportError("schedule package is required for --schedule option")

        schedule.every().week.do(
            reclassify,
            Path(args.dataset),
            Path(args.model_dir),
            args.clip_len,
            args.rename,
            args.summary,
        )
        while True:  # pragma: no cover - scheduling loop
            schedule.run_pending()
            time.sleep(60)
    else:
        reclassify(Path(args.dataset), Path(args.model_dir), args.clip_len, args.rename, args.summary)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
