import cv2
import numpy as np
import argparse


def format_time(seconds: float) -> str:
    """Return time in HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def detect_motion(video_path: str, motion_threshold: float = 25.0, min_duration: float = 0.5):
    """Detect high motion segments in a video.

    Parameters
    ----------
    video_path: str
        Path to video file.
    motion_threshold: float
        Average pixel difference (0-255) to trigger motion detection.
    min_duration: float
        Minimum duration of motion segment to record in seconds.

    Returns
    -------
    list of tuple(float, float)
        List of (start_seconds, end_seconds) for detected motion segments.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ret, prev = cap.read()
    if not ret:
        raise IOError("Unable to read frames from video")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_idx = 1
    motion_start = None
    segments = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        diff_val = float(np.mean(diff))

        if diff_val >= motion_threshold:
            if motion_start is None:
                motion_start = (frame_idx - 1) / fps
        else:
            if motion_start is not None:
                end_time = frame_idx / fps
                if end_time - motion_start >= min_duration:
                    segments.append((motion_start, end_time))
                motion_start = None

        prev_gray = gray
        frame_idx += 1

    if motion_start is not None:
        end_time = frame_idx / fps
        if end_time - motion_start >= min_duration:
            segments.append((motion_start, end_time))

    cap.release()
    return segments


def main():
    parser = argparse.ArgumentParser(description="Detect high-motion segments in a video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--threshold", type=float, default=25.0, help="Pixel difference to treat as motion")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Minimum motion duration in seconds")
    args = parser.parse_args()

    segments = detect_motion(args.video, args.threshold, args.min_duration)
    for start, end in segments:
        print(f"{format_time(start)} - {format_time(end)}")


if __name__ == "__main__":
    main()
