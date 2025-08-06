"""Play classifier using YOLOv5.

This module provides a simple wrapper over a YOLOv5 model to detect
football play events such as a touchdown (player moving quickly toward
the bottom of the frame) or a player exiting the frame at high speed.

The implementation relies on either OpenCV or PyTorch for image
processing. The model is expected to be a standard YOLOv5 model
trained to detect players. It can be loaded from a local path or via
``torch.hub``.
"""

from typing import List, Tuple

import numpy as np
import argparse
import json
import shutil
import csv
from pathlib import Path

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import torch
    from torchvision import models, transforms
except Exception:  # pragma: no cover - optional dependency
    torch = None


class PlayClassifier:
    """Classifies sequences of frames for touchdown-like events."""

    def __init__(self, model_path: str = "yolov5s.pt", device: str = "cpu") -> None:
        """Load a YOLOv5 model.

        Parameters
        ----------
        model_path:
            Path to the YOLOv5 model weights. ``torch.hub`` will be used
            to fetch the model if ``model_path`` points to a known name
            like ``yolov5s.pt``.
        device:
            Device string understood by ``torch`` (e.g. ``"cpu"`` or
            ``"cuda"``).
        """
        if torch is None:
            raise ImportError("PyTorch is required for PlayClassifier")

        # load model via torch.hub (from ultralytics) or local path
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=model_path,
            force_reload=False,
            trust_repo=True,
        ).to(device)
        self.model.eval()
        self.device = device

    def _detect_player(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Run YOLOv5 on a single frame and return the largest box.

        Returns
        -------
        (x1, y1, x2, y2):
            Coordinates of the detected player's bounding box. If no
            player is detected, ``(-1, -1, -1, -1)`` is returned.
        """
        if frame is None:
            return (-1, -1, -1, -1)

        # convert BGR frame to RGB as expected by YOLO
        img = frame[:, :, ::-1]
        results = self.model(img)
        if results is None or len(results.xyxy[0]) == 0:
            return (-1, -1, -1, -1)

        boxes = results.xyxy[0][:, :4].cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        return tuple(map(int, boxes[idx]))  # type: ignore

    def classify_sequence(
        self,
        frames: List[np.ndarray],
        speed_threshold: float = 20.0,
        edge_fraction: float = 0.1,
    ) -> List[Tuple[str, int, Tuple[int, int]]]:
        """Analyze frames and detect events.

        Parameters
        ----------
        frames:
            List of frames ordered in time.
        speed_threshold:
            Pixel distance per frame considered "fast".
        edge_fraction:
            Fraction of the frame (0-1) considered near the edge.

        Returns
        -------
        List of tuples ``(event, frame_index, center)`` where ``event`` is
        either ``"touchdown"`` or ``"exit"``.
        """
        events = []
        last_center = None
        last_index = 0
        for i, frame in enumerate(frames):
            box = self._detect_player(frame)
            if box[0] == -1:
                continue
            x1, y1, x2, y2 = box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if last_center is not None:
                dist = np.hypot(center[0] - last_center[0], center[1] - last_center[1])
                dt = i - last_index
                speed = dist / float(dt) if dt > 0 else 0.0
                h, w = frame.shape[:2]
                near_edge = (
                    center[0] < edge_fraction * w
                    or center[0] > (1.0 - edge_fraction) * w
                    or center[1] < edge_fraction * h
                    or center[1] > (1.0 - edge_fraction) * h
                )
                if speed > speed_threshold and near_edge:
                    events.append(("exit", i, center))
                if speed > speed_threshold and center[1] > (1.0 - edge_fraction) * h:
                    events.append(("touchdown", i, center))
            last_center = center
            last_index = i
        return events

    def classify_video(
        self,
        video_path: str,
        stride: int = 1,
        **kwargs,
    ) -> List[Tuple[str, int, Tuple[int, int]]]:
        """Load a video and classify it for events.

        Parameters
        ----------
        video_path:
            Path to the video file.
        stride:
            Use every ``stride``-th frame for analysis.
        **kwargs:
            Additional keyword arguments forwarded to ``classify_sequence``.
        """
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for video analysis")

        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame = cap.read()
        while success:
            frames.append(frame)
            for _ in range(stride - 1):
                success = cap.grab()
                if not success:
                    break
            success, frame = cap.read()
        cap.release()
        return self.classify_sequence(frames, **kwargs)





def classify_play(
    video_clip_path: str,
    metadata_json_path: str | None = None,
    model_path: str = "models/play_classifier/latest.pt",
    *,
    threshold: float = 0.6,
    log_uncertain: bool = False,
    log_file: str = "low_confidence_log.csv",
    review_dir: str = "manual_review",
) -> dict:
    """Classify a video clip into a play type using a pretrained model.

    Parameters
    ----------
    video_clip_path:
        Path to the video clip to classify.
    metadata_json_path:
        Optional path to accompanying metadata JSON.
    model_path:
        Path to the trained model weights.
    threshold:
        Confidence threshold below which a prediction will be flagged.
    log_uncertain:
        When ``True``, low-confidence predictions are logged and clips are
        saved for manual review.
    log_file:
        CSV file where low-confidence predictions are appended.
    review_dir:
        Directory where low-confidence clips are copied.
    """

    if torch is None or cv2 is None:
        raise ImportError("PyTorch and OpenCV are required for classify_play")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    label_map = checkpoint.get("label_map", {})
    inv_map = {v: k for k, v in label_map.items()}

    model = models.video.r3d_18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(label_map))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(str(video_clip_path))
    frames = []
    success, frame = cap.read()
    while success:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        tensor = transform(tensor)
        frames.append(tensor)
        success, frame = cap.read()
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_clip_path}")
    clip_len = 16
    while len(frames) < clip_len:
        frames.append(frames[-1].clone())
    clip = torch.stack(frames[:clip_len]).permute(1, 0, 2, 3).unsqueeze(0)

    with torch.no_grad():
        out = model(clip.to(device))
        probs = torch.softmax(out, dim=1)
        conf, pred = probs.max(1)

    label = inv_map.get(int(pred.item()), "unknown")
    confidence = float(conf.item())
    result = {"play_type": label, "confidence": confidence}

    if log_uncertain and confidence < threshold:
        review_path = Path(review_dir)
        review_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_clip_path, review_path / Path(video_clip_path).name)

        log_path = Path(log_file)
        exists = log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["clip", "play_type", "confidence"])
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "clip": Path(video_clip_path).name,
                    "play_type": label,
                    "confidence": confidence,
                }
            )

    if metadata_json_path:
        try:
            with open(metadata_json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            result["metadata"] = metadata
        except Exception:
            pass

    return result


def main() -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="Classify play clips")
    parser.add_argument("--folder", required=True, help="folder with video clips")
    parser.add_argument("--output", default="predictions.json", help="output JSON")
    parser.add_argument("--model", default="models/play_classifier/latest.pt")
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="Confidence threshold for review"
    )
    parser.add_argument(
        "--log-uncertain",
        action="store_true",
        help="Log and save clips when confidence is below threshold",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    results = []
    for clip in sorted(folder.glob("*.mp4")):
        meta = clip.with_suffix(".json")
        meta_path = str(meta) if meta.exists() else None
        pred = classify_play(
            str(clip),
            meta_path,
            args.model,
            threshold=args.threshold,
            log_uncertain=args.log_uncertain,
        )
        conf = pred["confidence"]
        results.append({"clip": clip.name, "play_type": pred["play_type"], "confidence": conf})
        if args.log_uncertain and conf < args.threshold:
            print(f"[REVIEW] {clip.name} ({conf:.2f})")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\u2705 Saved predictions to {args.output}")


__all__ = ["PlayClassifier", "classify_play"]


if __name__ == "__main__":
    main()
