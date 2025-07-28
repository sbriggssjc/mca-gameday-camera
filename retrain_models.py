import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def load_confirmed_jerseys() -> List[Dict[str, str]]:
    path = Path("/training/labels/confirmed_jerseys.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
    except Exception:
        return []
    for item in data:
        fname = item.get("filename")
        if fname:
            item["image_path"] = str(Path("/training/uncertain_jerseys") / fname)
    return data


def load_confirmed_play_types() -> List[Dict[str, str]]:
    path = Path("/training/labels/confirmed_play_types.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
    except Exception:
        return []
    for item in data:
        fname = item.get("filename")
        if fname:
            item["image_path"] = str(Path("/training/frames") / fname)
    return data


def summarize_jersey_labels(entries: List[Dict[str, str]]) -> None:
    counts: Dict[str, int] = {}
    for e in entries:
        num = str(e.get("jersey_number", ""))
        counts[num] = counts.get(num, 0) + 1
    print(f"{len(entries)} jersey images found")
    for num, cnt in sorted(
        counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]
    ):
        print(f"  {num}: {cnt}")


def summarize_play_labels(entries: List[Dict[str, str]]) -> None:
    counts: Dict[str, int] = {}
    for e in entries:
        label = e.get("play_type", "unknown")
        counts[label] = counts.get(label, 0) + 1
    print(f"{len(entries)} play examples found")
    for label, cnt in sorted(counts.items()):
        print(f"  {label}: {cnt}")


def retrain_jersey_ocr(dry_run: bool = False, summary_only: bool = False) -> None:
    entries = load_confirmed_jerseys()
    summarize_jersey_labels(entries)
    if summary_only:
        return
    dataset_dir = Path("/models/ocr/dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for e in entries:
        src = Path(e.get("image_path", ""))
        dst = dataset_dir / e.get("filename", "")
        if not src.is_file():
            continue
        shutil.copy(src, dst)
        with open(dst.with_suffix(".gt.txt"), "w", encoding="utf-8") as f:
            f.write(str(e.get("jersey_number", "")))
    print(f"\u2705 Dataset exported to {dataset_dir}")
    if dry_run:
        print("Dry run, skipping Tesseract training")
        return
    print(
        "Please run Tesseract training with the exported dataset to update the OCR model."
    )


def retrain_play_classifier(dry_run: bool = False, summary_only: bool = False) -> None:
    entries = load_confirmed_play_types()
    summarize_play_labels(entries)
    if summary_only:
        return
    try:
        import torch
        from torchvision import datasets, models, transforms
    except Exception:
        print("PyTorch or torchvision not available. Skipping training.")
        return
    dataset_root = Path("/models/play_recognition/dataset")
    dataset_root.mkdir(parents=True, exist_ok=True)
    for e in entries:
        src = Path(e.get("image_path", ""))
        label = e.get("play_type", "unknown")
        if not src.is_file():
            continue
        dest_dir = dataset_root / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest_dir / e.get("filename", ""))
    if dry_run:
        print("Dry run, dataset prepared in", dataset_root)
        return
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.ImageFolder(str(dataset_root), transform=transform)
    if len(ds) == 0:
        print("No training images found.")
        return
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, len(ds.classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(3):
        model.train()
        running = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        print(f"Epoch {epoch + 1}: loss {running / len(ds):.4f}")
    out_dir = Path("/models/play_recognition")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "classes": ds.classes},
        out_dir / "play_classifier.pt",
    )
    print(f"\u2705 Saved model to {out_dir / 'play_classifier.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain OCR and play recognition models"
    )
    parser.add_argument(
        "--target", choices=["jersey_ocr", "play_classifier"], required=True
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Prepare datasets without full training"
    )
    parser.add_argument(
        "--summary_only", action="store_true", help="Only print dataset summaries"
    )
    args = parser.parse_args()

    if args.target == "jersey_ocr":
        retrain_jersey_ocr(dry_run=args.dry_run, summary_only=args.summary_only)
    elif args.target == "play_classifier":
        retrain_play_classifier(dry_run=args.dry_run, summary_only=args.summary_only)


if __name__ == "__main__":
    main()
