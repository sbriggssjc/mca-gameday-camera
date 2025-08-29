from __future__ import annotations

import argparse, cv2, numpy as np
import os
from typing import List, Tuple

import torch
import torch.nn as nn  # <-- this was missing
from torchvision import models, transforms

if not torch.cuda.is_available():
    print("[warn] CUDA not available – running classifier on CPU")


class ToFloatNormalize(nn.Module):
    """Convert ``uint8`` tensor to float and normalise to ImageNet stats."""

    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x / 255.0
        return (x - self.mean) / self.std


def _load_model(ckpt: str, labels_path: str | None) -> Tuple[nn.Module, List[str]]:
    data = torch.load(ckpt, map_location="cpu")
    label_map = data.get("label_map")
    labels: List[str] = []
    if label_map:
        inv = {v: k for k, v in label_map.items()}
        labels = [inv[i] for i in range(len(inv))]
    elif labels_path and os.path.isfile(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [ln.strip() for ln in f if ln.strip()]
    if not labels:
        raise RuntimeError("Could not load labels for classifier")
    model = models.video.r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, labels


def _sample_frames(path: str, count: int = 8) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError("Could not determine frame count")
    start = int(total * 0.25)
    end = max(start + 1, int(total * 0.75))
    indices = np.linspace(start, end - 1, count).astype(int)
    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("Failed to sample frames from video")
    return frames


def _prep_batch(frames: List[np.ndarray]) -> torch.Tensor:
    transform = transforms.Compose([transforms.Resize((224, 224)), ToFloatNormalize()])
    clips = []
    for fr in frames:
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        tensor = transform(tensor)
        clip = tensor.unsqueeze(1).repeat(1, 16, 1, 1)
        clips.append(clip)
    batch = torch.stack(clips)
    return batch


def run(video: str, play_ckpt: str, play_labels: str | None) -> np.ndarray:
    model, labels = _load_model(play_ckpt, play_labels)
    frames = _sample_frames(video, 8)
    batch = _prep_batch(frames)
    with torch.no_grad():
        out = model(batch)
        probs = torch.softmax(out, dim=1).cpu().numpy()
    for i, pr in enumerate(probs):
        idx = pr.argsort()[::-1][:3]
        desc = ", ".join(f"{labels[j]}:{pr[j]:.3f}" for j in idx)
        print(f"frame {i}: {desc}")
    avg = probs.mean(axis=0)
    idx = avg.argsort()[::-1][:3]
    desc = ", ".join(f"{labels[j]}:{avg[j]:.3f}" for j in idx)
    print(f"average: {desc}")
    if all(pr.max() < 0.05 for pr in probs):
        raise SystemExit(1)
    return probs


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--play-ckpt", default="models/play_classifier/latest.pt")
    p.add_argument("--play-labels", default="models/play_classifier/labels.txt")
    p.add_argument("--formation-ckpt", default="models/formation/latest.pt")
    p.add_argument("--formation-labels", default="models/formation/labels.txt")
    args = p.parse_args(argv)
    run(args.video, args.play_ckpt, args.play_labels)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
