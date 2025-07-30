from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

try:
    import cv2  # type: ignore
    import torch
except Exception:  # pragma: no cover - optional
    cv2 = None
    torch = None


class HighlightClipDataset(torch.utils.data.Dataset):
    """Simple dataset for highlight video clips."""

    def __init__(self, csv_file: str | Path, clip_len: int = 16, transform=None) -> None:
        if torch is None or cv2 is None:
            raise ImportError("PyTorch and OpenCV are required for HighlightClipDataset")
        self.entries: List[dict] = []
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append(row)
        self.clip_len = clip_len
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def _read_frames(self, path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(path))
        frames: List[torch.Tensor] = []
        success, frame = cap.read()
        while success and len(frames) < self.clip_len:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
            if self.transform:
                tensor = self.transform(tensor)
            frames.append(tensor)
            success, frame = cap.read()
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames read from {path}")
        while len(frames) < self.clip_len:
            frames.append(frames[-1].clone())
        return torch.stack(frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        item = self.entries[idx]
        frames = self._read_frames(Path(item["filepath"]))
        label = item["label"]
        return frames, label


__all__ = ["HighlightClipDataset"]
