"""Overlay scoreboard data onto video frames and stream via ffmpeg."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import cv2

from scoreboard_reader import ScoreboardReader, ScoreboardState


def ensure_ffmpeg() -> str:
    """Return path to ffmpeg or raise if missing."""
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg is not installed or not in PATH")
    return path


def select_codec() -> str:
    """Choose a hardware-accelerated codec if available."""
    try:
        output = subprocess.check_output(["ffmpeg", "-encoders"], text=True)
        for codec in ("h264_nvmpi", "h264_nvv4l2enc"):
            if codec in output:
                return codec
    except Exception:
        pass
    return "libx264"


class OverlayEngine:
    """Render scoreboard information on frames."""

    def __init__(self, *, font_scale: float = 1.0, color: tuple[int, int, int] = (255, 255, 255), position: str = "left") -> None:
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.color = color
        self.thickness = 2
        self.position = position

    def draw(self, frame, state: ScoreboardState) -> None:
        """Draw the overlay in-place."""
        text = f"Home {state.home} - {state.away} Away  Q{state.quarter}  {state.clock}"
        if state.down is not None and state.distance is not None:
            text += f"  {state.down} & {state.distance}"
        if self.position == "center":
            text_size, _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
            x = int((frame.shape[1] - text_size[0]) / 2)
        else:
            x = 20
        cv2.putText(frame, text, (x, 40), self.font, self.font_scale, self.color, self.thickness, cv2.LINE_AA)


def load_state_from_json(path: str, fallback: ScoreboardState) -> ScoreboardState:
    """Load scoreboard state from a JSON file, returning the fallback if unavailable."""
    if not os.path.exists(path):
        return fallback
    try:
        with open(path) as f:
            data = json.load(f)
        return ScoreboardState(
            home=int(data.get("home", fallback.home)),
            away=int(data.get("away", fallback.away)),
            quarter=int(data.get("quarter", fallback.quarter)),
            clock=str(data.get("clock", fallback.clock)),
            down=(int(data["down"]) if data.get("down") is not None else fallback.down),
            distance=(int(data["distance"]) if data.get("distance") is not None else fallback.distance),
        )
    except Exception:
        return fallback


def build_ffmpeg_command(url: str, size: tuple[int, int], fps: float) -> list[str]:
    """Return command list for launching ffmpeg."""
    width, height = size
    return [
        ensure_ffmpeg(),
        "-loglevel",
        "verbose",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-c:v",
        select_codec(),
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "flv",
        url,
    ]


def stream(device: int, rtmp_url: str, *, json_path: str = "game_state.json", position: str = "left") -> None:
    """Stream camera frames with overlay to the provided RTMP URL."""
    ensure_ffmpeg()

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open capture device {device}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    log_dir = Path("livestream_logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    lf = log_file.open("w")
    cmd = build_ffmpeg_command(rtmp_url, (width, height), fps)
    print("Running FFmpeg command:", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _reader(pipe, logf):
        for line in pipe:
            print(line, end="")
            logf.write(line)

    thread = threading.Thread(target=_reader, args=(process.stdout, lf), daemon=True)
    thread.start()
    overlay = OverlayEngine(position=position)
    manual_reader = ScoreboardReader()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            manual_state = manual_reader.update(frame)
            state = load_state_from_json(json_path, manual_state)
            overlay.draw(frame, state)
            process.stdin.write(frame.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if process.stdin:
            process.stdin.close()
        ret = process.wait()
        thread.join()
        lf.write(f"\nffmpeg exited with code {ret}\n")
        lf.close()
        if ret != 0:
            print("FFmpeg exited with error:", ret)


def overlay_file(input_path: str, output_path: str, *, json_path: str | None = None, position: str = "left") -> None:
    """Overlay scoreboard on a video file. Prompts for manual input if needed."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    overlay = OverlayEngine(position=position)
    manual_reader = ScoreboardReader()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        manual_state = manual_reader.update(frame)
        state = manual_state
        if json_path:
            state = load_state_from_json(json_path, manual_state)
        overlay.draw(frame, state)
        writer.write(frame)

    cap.release()
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay scoreboard data on video")
    parser.add_argument("--rtmp-url", help="RTMP URL for live streaming")
    parser.add_argument("--device", type=int, default=0, help="video capture device index")
    parser.add_argument("--video", help="video file to overlay after the game")
    parser.add_argument("--output", help="output file when processing a video")
    parser.add_argument("--json", default="game_state.json", help="path to game_state.json")
    parser.add_argument("--position", choices=["left", "center"], default="left", help="overlay position")
    args = parser.parse_args()

    if args.rtmp_url:
        stream(args.device, args.rtmp_url, json_path=args.json, position=args.position)
    elif args.video and args.output:
        overlay_file(args.video, args.output, json_path=args.json if os.path.exists(args.json) else None, position=args.position)
    else:
        parser.print_help()
