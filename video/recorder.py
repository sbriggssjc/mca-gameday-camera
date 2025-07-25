import os
import time
import argparse

import cv2


class VideoRecorder:
    """Record 720p video to disk with optional highlight clips."""

    def __init__(self, output_path: str, clip_dir: str = "clips", fps: int = 30):
        self.output_path = output_path
        self.clip_dir = clip_dir
        self.fps = fps
        os.makedirs(self.clip_dir, exist_ok=True)

        # initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.full_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (1280, 720))

        self.clip_writer = None
        self.clip_end_time = None
        self.next_clip_id = 1

    def start_highlight(self):
        """Begin writing a 30-second highlight clip."""
        if self.clip_writer is not None:
            return
        clip_name = os.path.join(self.clip_dir, f"clip_{self.next_clip_id:03d}.mp4")
        self.next_clip_id += 1
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.clip_writer = cv2.VideoWriter(clip_name, fourcc, self.fps, (1280, 720))
        self.clip_end_time = time.time() + 30
        print(f"Started highlight recording: {clip_name}")

    def _write_frame(self, frame):
        self.full_writer.write(frame)
        if self.clip_writer:
            self.clip_writer.write(frame)
            if time.time() >= self.clip_end_time:
                print("Highlight clip finished")
                self.clip_writer.release()
                self.clip_writer = None

    def close(self):
        if self.clip_writer:
            self.clip_writer.release()
        self.full_writer.release()
        self.cap.release()

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                self._write_frame(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("h"):
                    self.start_highlight()
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description="Record video with optional highlights")
    parser.add_argument("--output", default="full_recording.mp4", help="Path for full length video")
    parser.add_argument("--clips", default="clips", help="Directory for highlight clips")
    args = parser.parse_args()

    recorder = VideoRecorder(args.output, args.clips)
    recorder.run()


if __name__ == "__main__":
    main()
