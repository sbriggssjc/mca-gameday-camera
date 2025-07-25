import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
import numpy as np
import time
from datetime import datetime
import subprocess


def open_writer(path: str, fps: float, size: tuple[int, int]):
    """Open H.264 writer, fallback to MJPG if unavailable."""
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        print("avc1 codec not available, falling back to MJPG")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, fps, size)
    return writer


def monitor(device: str = "/dev/video0", upload: bool = True) -> None:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera {device}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Failed to read initial frame")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    recording = False
    writer = None
    record_end = 0.0
    clip_path = ""
    no_motion_start = time.time()
    motion_threshold = 25.0

    print("Monitoring for motion... Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed, stopping.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff_val = float(np.mean(cv2.absdiff(prev_gray, gray)))
            now = time.time()

            if diff_val >= motion_threshold:
                if not recording and no_motion_start is not None and now - no_motion_start >= 5:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    clip_path = f"highlight_{timestamp}.mp4"
                    writer = open_writer(clip_path, fps, (1280, 720))
                    if not writer.isOpened():
                        print("Failed to open video writer")
                        break
                    recording = True
                    record_end = now + 10
                    print(f"Motion detected. Recording started: {clip_path}")
                no_motion_start = None
            else:
                if no_motion_start is None:
                    no_motion_start = now

            if recording:
                writer.write(frame)
                if now >= record_end:
                    writer.release()
                    recording = False
                    print(f"Clip saved: {clip_path}")
                    if upload:
                        result = subprocess.run(
                            ["rclone", "copy", clip_path, "gdrive:/MCA/GameDayHighlights/"],
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode != 0:
                            print("Upload failed:")
                            print(result.stderr.strip())
                        else:
                            print("Upload successful.")
                    no_motion_start = now

            prev_gray = gray
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        if writer and writer.isOpened():
            writer.release()
        print("Stopped.")


if __name__ == "__main__":
    monitor()
