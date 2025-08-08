# mca-gameday-camera

This repository contains utilities for tracking play participation during a game.

Large video recordings (`.mp4`) are saved in the `video/` folder but individual recording files are ignored by Git. Use `upload_to_drive.py` to sync these videos to Google Drive instead of committing them.

## Processing uploaded game film

Place a video inside `video/manual_uploads/` and run:

```bash
python run_uploaded_film.py --video my_game.mp4 --purge_after True
```

The clip is analyzed locally and both the raw video and summary JSON are uploaded to Google Drive. The local video is removed only after a successful upload while the logs remain under `output/summary/` and `output/manual_logs/`.

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

## stream_to_youtube.py

`stream_to_youtube.py` streams a video device (default `/dev/video0`) to
YouTube using `ffmpeg`. Set the `VIDEO_DEVICE` environment variable if you need
to use a different camera. Place your YouTube stream key in a `.env` file:

```ini
YOUTUBE_STREAM_KEY=your_actual_stream_key
```

You can also provide the key at runtime with `--stream-key`. Logs are written to the `livestream_logs` folder and the script will
automatically restart `ffmpeg` if it exits unexpectedly.

Run it with:

```bash
python stream_to_youtube.py
```

The default settings use the software `libx264` encoder at
1920x1080 and 30fps with a bitrate around **9&nbsp;Mbps**.
Output is written with `tee` so a local MP4 recording is saved
alongside the live RTMP stream.

If `GDRIVE_FOLDER_ID` is set, the MP4 and a matching
`*_play_log.csv` are uploaded to that Drive folder after streaming
finishes. Set `GDRIVE_USE_GAME_FOLDER=1` to place both files in a
dedicated subfolder named after the game timestamp.

Additional options:

```bash
python stream_to_youtube.py --output-size 426x240 --debug
```

## Requirements

- Python 3
- [OpenCV](https://opencv.org/)
- [ffmpeg](https://ffmpeg.org/)

Install FFmpeg on Jetson with:

```bash
sudo apt-get update && sudo apt-get install ffmpeg
```

For best performance on Jetson devices, build FFmpeg with the Jetson
accelerated encoders (`h264_nvmpi` or `h264_nvv4l2enc`). The streaming
scripts automatically fall back to `libx264` when these encoders are
unavailable.

This repository contains simple utilities for analyzing football plays.

## Modules

- `play_classifier.py` – includes the `PlayClassifier` class for touchdown detection
  and a `classify_play` function to label short clips using a pretrained video model.
  Run `python play_classifier.py --folder clips/ --output predictions.json` to classify
  a directory of clips.
- `record_video.py` – records 426x240 video from /dev/video0 to output.mp4
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

You can also build a dataset from highlight clips:

```bash
python build_highlight_dataset.py highlights/ dataset/
```

This copies the clips into `dataset` and creates `dataset/metadata.csv`:

```csv
filepath,label,quarter,time,player
dataset/TD_JaxonBrunner_Jet_Sweep_Q2_05m12s.mp4,Jet Sweep,Q2,05:12,JaxonBrunner
```

The `HighlightClipDataset` class in `highlight_dataset.py` loads these clips as
PyTorch tensors for training models.

## reclassify_old_clips.py

`reclassify_old_clips.py` runs the latest play classifier on your existing
highlight clips and updates their labels if the prediction changes. Updates are
appended to `training/logs/learning_log.json`.

```bash
python reclassify_old_clips.py dataset/metadata.csv --model_dir models/play_classifier
```

Add `--schedule` to run weekly (requires the `schedule` package) or create a
cron entry:

```cron
0 3 * * 0 /usr/bin/python /path/to/mca-gameday-camera/reclassify_old_clips.py dataset/metadata.csv --model_dir models/play_classifier
```

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

## upload_to_drive.py

`upload_to_drive.py` sends finished recordings to Google Drive using
[PyDrive](https://github.com/googledrive/PyDrive). The first run requires a
`client_secrets.json` OAuth file in this directory. After authenticating,
credentials are stored in `drive_token.json` so future runs reuse them.
Set the destination folder ID with the `GDRIVE_FOLDER_ID` environment variable:

```bash
export GDRIVE_FOLDER_ID=your_folder_id
python upload_to_drive.py video/game_20250727_080156.mp4
```

To automate uploads, run `upload_daily.sh` via cron:

```cron
0 2 * * * /path/to/mca-gameday-camera/upload_daily.sh
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

## Recording Storage

Raw recordings can quickly exceed GitHub's size limits. Do **not** commit any of the files in the `video/` directory or other `.mp4` footage. Instead use `sync_to_drive.py` to upload clips to your Drive folder and keep the repository clean.

If you want to store large assets with Git, install Git LFS and track MP4 files:

```bash
git lfs install
git lfs track "*.mp4"
git add .gitattributes
```

## generate_scouting_report.py

Create a scouting report summarizing an opponent's play tendencies. The script
expects a `scouting_data.csv` file with columns:

```
game_date,opponent,offense,formation,label,down,quarter,yards_gained
```

Run it with the opponent name to produce a PDF or text report under
`analysis/`:

```bash
python generate_scouting_report.py "Victory Christian"
```

If the optional `fpdf` package is installed the output will be a PDF,
otherwise a plain text file is generated.

## generate_hudl_csv.py

Export labeled clips to a HUDL-compatible CSV. The script reads from
`highlight_log.csv` or `scouting_data.csv` and writes a file under
`hudl_export/`.

```bash
python generate_hudl_csv.py --week 3 --opponent "Victory Christian"
```

Use `--player 23` to limit rows to a specific jersey number.
