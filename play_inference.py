import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Dict, List

try:
    import cv2  # type: ignore
    import torch
    from torch import nn
    from torchvision import models, transforms
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    torch = None


class ToFloatNormalize(nn.Module):
    """Convert ``uint8`` tensor to float and normalize to ImageNet stats."""

    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x / 255.0
        return (x - self.mean) / self.std


def seconds_to_time(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def segment_video(video: str, out_dir: Path, segment_time: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        video,
        "-f",
        "segment",
        "-segment_time",
        str(segment_time),
        "-c",
        "copy",
        str(out_dir / "clip_%03d.mp4"),
    ]
    subprocess.run(cmd, check=True)


def load_model(checkpoint: str, device: torch.device) -> tuple[nn.Module, Dict[int, str]]:
    data = torch.load(checkpoint, map_location=device)
    label_map = data.get("label_map", {})
    inv_map = {v: k for k, v in label_map.items()}
    model = models.video.r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(label_map))
    model.load_state_dict(data["model_state"])
    model = model.to(device)
    model.eval()
    return model, inv_map


def read_clip(path: Path, clip_len: int, transform) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    frames: List[torch.Tensor] = []
    success, frame = cap.read()
    while success and len(frames) < clip_len:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        if transform:
            tensor = transform(tensor)
        frames.append(tensor)
        success, frame = cap.read()
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    while len(frames) < clip_len:
        frames.append(frames[-1].clone())
    clip = torch.stack(frames)  # (T, C, H, W)
    return clip.permute(1, 0, 2, 3)  # (C, T, H, W)


def run_inference(video: str, checkpoint: str, segment_time: int, clips_dir: str, output_csv: str) -> None:
    if cv2 is None or torch is None:
        raise ImportError("PyTorch and OpenCV are required for inference")

    clip_dir = Path(clips_dir)
    segment_video(video, clip_dir, segment_time)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, inv_map = load_model(checkpoint, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        ToFloatNormalize(),
    ])

    log_rows = []
    clip_files = sorted(clip_dir.glob("clip_*.mp4"))
    for idx, clip_path in enumerate(clip_files):
        start = idx * segment_time
        end = start + segment_time
        clip = read_clip(clip_path, 16, transform)
        clip = clip.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(clip)
        pred = out.argmax(1).item()
        label = inv_map.get(pred, "unknown")
        log_rows.append({
            "start_time": seconds_to_time(start),
            "end_time": seconds_to_time(end),
            "label": label,
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start_time", "end_time", "label"])
        writer.writeheader()
        writer.writerows(log_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run play classifier on full game video")
    parser.add_argument("video", help="Full game video file")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--segment_time", type=int, default=8, help="Clip length in seconds")
    parser.add_argument("--clips_dir", default="plays", help="Directory for temporary clips")
    parser.add_argument("--output_csv", default="play_log.csv", help="CSV file for results")
    args = parser.parse_args()

    run_inference(args.video, args.model, args.segment_time, args.clips_dir, args.output_csv)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
