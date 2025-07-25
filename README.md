# mca-gameday-camera

This repository contains utilities for tracking play participation during a game.

## play_count_tracker.py

`play_count_tracker.py` is a simple command line tool for recording which players were on the field for each play. After each play, type the jersey numbers (1-17) separated by spaces. Enter `q` to finish. A log is written to `play_log.csv` and the final play counts are printed with alerts for any player that participated in fewer than seven plays.

### Usage

```bash
python play_count_tracker.py
```
This repository contains tools for processing sports game footage. The `motion_detector.py` script scans a video and prints timecodes for periods of high motion. These timecodes are useful for extracting highlight clips from a full game recording.

## Usage

```bash
python motion_detector.py path/to/video.mp4
```

You can adjust the detection sensitivity using `--threshold` and minimum segment length with `--min-duration`.
This repository provides a simple script for streaming a camera feed to
YouTube Live using RTMP. Frames are captured with OpenCV and piped to
`ffmpeg` for encoding and upload.

## Usage

```
python streamer.py <youtube_rtmp_url> [device_index]
```

Replace `<youtube_rtmp_url>` with the RTMP URL and stream key supplied by
YouTube. The optional `device_index` selects which local camera to use
(the default is `0`). Press `Ctrl+C` to end the stream.

## Requirements

- Python 3
- [OpenCV](https://opencv.org/)
- [ffmpeg](https://ffmpeg.org/)

This repository contains simple utilities for analyzing football plays.

## Modules

- `play_classifier.py` – provides the `PlayClassifier` class which wraps a
  YOLOv5 model to detect touchdown-like movements or fast exits of a
  player from the frame.
- `record_video.py` – records 1280x720 video from /dev/video0 to output.mp4
- `highlight_recorder.py` – automatically captures 10-second clips when motion is detected

## update_code.sh

Run `update_code.sh` to pull the latest changes from the remote `main` branch. The script handles errors like missing Git or network issues and prints whether new code was retrieved or if the repository was already current.
