import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
import time


def open_writer(path: str, fps: float, size: tuple[int, int]):
    """Attempt to open avc1 (H.264) writer and fallback to MJPG."""
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        print("avc1 codec not available, falling back to MJPG")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fourcc, fps, size)
    return writer


def record(device: str = "/dev/video0", duration: int = 30) -> None:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera {device}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    writer = open_writer("output.mp4", fps, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer")

    print("Recording started. Press Ctrl+C to stop.")
    start = time.time()
    next_report = 5
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed, stopping.")
                break
            writer.write(frame)
            elapsed = time.time() - start
            if elapsed >= next_report:
                print(f"{int(elapsed)} seconds elapsed...")
                next_report += 5
            if elapsed >= duration:
                break
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    finally:
        cap.release()
        writer.release()
        print("Recording complete.")


if __name__ == "__main__":
    record()
