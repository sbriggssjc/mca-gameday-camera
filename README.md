# mca-gameday-camera

This repository contains utilities for tracking play participation during a game.

## play_count_tracker.py

`play_count_tracker.py` is a command line tool for recording which players were on the field for each play. It now supports optional in-game alerts, SMS notifications and quarter summaries. After each play, type the jersey numbers separated by spaces. Enter `q` to finish. A log is written to `jersey_counts.csv` and the final play counts are printed with color-coded warnings for any player under the threshold.

### Usage

```bash
python play_count_tracker.py --voice --quarters
```
This repository contains tools for processing sports game footage. The `motion_detector.py` script scans a video and prints timecodes for periods of high motion. These timecodes are useful for extracting highlight clips from a full game recording.

## Usage

```bash
python motion_detector.py path/to/video.mp4
```

You can adjust the detection sensitivity using `--threshold` and minimum segment length with `--min-duration`.
This repository provides simple scripts for streaming and recording a camera
feed. Frames can be captured with OpenCV and piped to `ffmpeg` for encoding and
upload.

## Usage

```
python streamer.py <youtube_rtmp_url> [device_index]
```

Replace `<youtube_rtmp_url>` with the RTMP URL and stream key supplied by
YouTube. The optional `device_index` selects which local camera to use
(the default is `0`). Press `Ctrl+C` to end the stream.

## stream_to_youtube.py

`stream_to_youtube.py` streams `/dev/video0` to YouTube using `ffmpeg`. The
YouTube RTMP URL should be placed in a `.env` file:

```ini
YOUTUBE_RTMP_URL=rtmp://a.rtmp.youtube.com/live2/YOUR_STREAM_KEY
```

Logs are written to the `livestream_logs` folder and the script will
automatically restart `ffmpeg` if it exits unexpectedly.

Run it with:

```bash
python stream_to_youtube.py
```

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
- `play_recognizer.py` – identifies plays based on formations in `mca_playbook.json` and writes results to `play_log.json`.
- `practice_trainer.py` – analyzes labeled practice clips and stores motion
  statistics in `training_set.json` for use by `play_recognizer.py`.
```bash
python play_recognizer.py path/to/game.mp4 --playbook mca_playbook.json --output play_log.csv
```
You can generate training data from practice clips:

```bash
python practice_trainer.py practice_clips/ --output training_set.json
```
Then supply `--training-data training_set.json` when running
`play_recognizer.py` to bias recognition toward those patterns.

## update_code.sh

Run `update_code.sh` to pull the latest changes from the remote `main` branch. The script handles errors like missing Git or network issues and prints whether new code was retrieved or if the repository was already current.

## gameday.sh

`gameday.sh` updates the repository and starts `highlight_recorder.py`.
You can place the script on the Desktop, make it executable with `chmod +x`,
and then right-click and select **Allow Launching** to use it like a shortcut.

## start_gameday.bat

Windows users can run `start_gameday.bat` to launch livestreaming,
recording and play tracking. The script loads the RTMP URL from `.env`,
checks that the camera is connected and then starts several Python
processes in separate terminal windows.

`launch_gameday.bat` provides a minimal launcher that only starts the livestream
and the play tracker:

```bat
start cmd /k "python stream_to_youtube.py"
start cmd /k "python play_count_tracker.py"
```

## youtube_uploader.py

`youtube_uploader.py` uploads a video file to your YouTube channel. The first
run uses OAuth2 to store credentials in `token.json`.

### One-time setup

1. Enable the **YouTube Data API v3** for your Google Cloud project.
2. Create OAuth client credentials (Desktop) and download `client_secrets.json`.
3. Place `client_secrets.json` in this folder.

### Usage

```bash
python youtube_uploader.py --file path/to/video.mp4 --title "My Title" \
    --description "Short description" --privacy public
```

## install_firefox_esr.py

`install_firefox_esr.py` downloads and extracts the latest Firefox ESR build for ARM64 Linux. It automatically detects the newest version from Mozilla's release archive and places Firefox in your home directory.

```bash
python install_firefox_esr.py
```

The script prints progress messages and optionally launches Firefox when done. It requires the `requests` package.

## assignment_analyzer.py

`assignment_analyzer.py` is a stub for rating player assignments in recorded clips.
It relies on `ai_detector.detect_jerseys` to find jersey numbers in each frame
and appends the results to `player_ratings.csv`.

```bash
python assignment_analyzer.py path/to/clip.mp4 --playbook playbook.json
```

The optional JSON playbook maps jersey numbers to assignments. Real jersey
detection and movement analysis are not implemented in this repository.
