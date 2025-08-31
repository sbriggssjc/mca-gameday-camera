import argparse
import csv
<<<<<<< HEAD
import logging
=======
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
import os
import subprocess
from pathlib import Path
from typing import Dict, List

try:
    import cv2  # type: ignore
    import torch
    from torch import nn
    from torchvision import models, transforms
<<<<<<< HEAD
    from torchvision.transforms import InterpolationMode
=======
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    torch = None


<<<<<<< HEAD
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
log = logging.getLogger(__name__)
=======
class ToFloatNormalize(nn.Module):
    """Convert ``uint8`` tensor to float and normalize to ImageNet stats."""

    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x / 255.0
        return (x - self.mean) / self.std
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa


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
<<<<<<< HEAD
        frame = frame.astype("float32") / 255.0
        tensor = torch.from_numpy(frame).permute(2, 0, 1)
=======
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
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
<<<<<<< HEAD
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.Normalize(mean=MEAN, std=STD),
=======
        transforms.Resize((224, 224)),
        ToFloatNormalize(),
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
    ])

    log_rows = []
    clip_files = sorted(clip_dir.glob("clip_*.mp4"))
<<<<<<< HEAD
    logged_stats = False
=======
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
    for idx, clip_path in enumerate(clip_files):
        start = idx * segment_time
        end = start + segment_time
        clip = read_clip(clip_path, 16, transform)
<<<<<<< HEAD
        if not logged_stats:
            log.info("input clip stats mean=%.4f std=%.4f", clip.mean().item(), clip.std().item())
            logged_stats = True
=======
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
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
<<<<<<< HEAD
    logging.basicConfig(level=logging.INFO)
=======
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
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
