"""Train a video-based play classification model."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from highlight_dataset import HighlightClipDataset


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
log = logging.getLogger(__name__)


class PlayVideoDataset(HighlightClipDataset):
    """Dataset returning clips and integer labels."""

    def __init__(self, csv_file: str | Path, label_map: Dict[str, int], clip_len: int = 16) -> None:
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
        super().__init__(csv_file, clip_len=clip_len, transform=transform)
        self.label_map = label_map

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        clip, label = super().__getitem__(idx)
        clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)
        return clip, self.label_map[label]


def collate_batch(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    clips, labels = zip(*batch)
    return torch.stack(clips), torch.tensor(labels)


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    logged_stats = False
    for clips, labels in loader:
        if not logged_stats:
            log.info("input batch stats mean=%.4f std=%.4f", clips.mean().item(), clips.std().item())
            logged_stats = True
        clips = clips.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * clips.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / total if total else 0.0
    print(f"train loss {running_loss / total:.4f} acc {acc:.2%}")
    return running_loss / total


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train play classification model")
    parser.add_argument("csv", help="metadata CSV with filepath and label columns")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", default="models/play_classifier")
    args = parser.parse_args()

    # discover labels
    labels: List[str] = []
    with open(args.csv) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                labels.append(parts[1])
    label_map = {lbl: i for i, lbl in enumerate(sorted(set(labels)))}

    dataset = PlayVideoDataset(args.csv, label_map, clip_len=args.clip_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    model = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, len(label_map))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_epoch(model, loader, criterion, optimizer, device)
        ckpt_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
        torch.save({"model_state": model.state_dict(), "label_map": label_map}, ckpt_path)
        print(f"\u2705 Saved {ckpt_path}")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
