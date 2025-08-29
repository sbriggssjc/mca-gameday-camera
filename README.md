# mca-gameday-camera

This repository contains utilities for tracking play participation during a game.

Large video recordings (`.mp4`) are saved in the `video/` folder but individual recording files are ignored by Git. Use `upload_to_drive.py` to sync these videos to Google Drive instead of committing them.

This project uses a single playbook at `playbooks/mca_5th_playbook.json`.

## Robust game-day capture

`gameday_capture.py` is a single-entry CLI that always records a local MP4 and
streams to YouTube when possible.  A thin wrapper is provided under
`scripts/gameday.sh` which loads `.env` and launches the capture.

### One-liner

```bash
scripts/gameday.sh
```

### Configure defaults via `.env`

```
YOUTUBE_RTMP_URL=rtmps://a.rtmps.youtube.com/live2/<key>
VIDEO_DEV=/dev/video0
PULSE_DEV=hw:1,0            # or Pulse source name
RES=1280x720
FPS=30
```

### Useful test modes

```bash
python3 gameday_capture.py --probe-only
python3 gameday_capture.py --duration 30 --local-only
```

## Game-day one-liners

```
USE_LOUDNORM=true ./gameday --audio-source pulse          # auto-gain to ~−16 dB
EXTRA_GAIN_DB=4 ./gameday --audio-source pulse            # manual tweak
./gameday --dry-run | less                                # inspect command
```

## Quick Start (YouTube Live)

```bash
# 0) First time: check system basics
scripts/doctor.sh

# 1) Set your stream key (NO angle brackets, no spaces)
export YT_RTMP_URL='rtmps://a.rtmps.youtube.com/live2/<your_key>'
# (``YOUTUBE_RTMP_URL`` is also accepted)

# 2) Optional: override devices (else put them in config/gameday.json)
export VIDEO_DEV=/dev/video0
export PULSE_DEV='alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_XXXXXXXX-00.mono-fallback'

# 3) Run
./gameday
```

If 443/rtmps is flaky, try:

rtmps://b.rtmps.youtube.com/live2/<key>

If port 1935 is open and you prefer RTMP:

rtmp://a.rtmp.youtube.com/live2/<key>

Notes

The launcher prints a one-line “Launch -> …” status to stderr and emits JSON config to stdout internally. If it says missing or invalid RTMP URL, fix your key.

If YouTube shows “No data” or a very low bitrate, verify network, try b.rtmps, or switch to rtmp:// if 1935 is open.

We avoid aresample min_comp/max_comp entirely for compatibility.

MJPEG → H.264 path adds in_range=jpeg:out_range=tv to prevent washed/incorrect levels on YT.

---

## Optional: test generators (keep for debugging)

```bash
ffmpeg -hide_banner -loglevel info -re \
  -f lavfi -i testsrc2=size=1280x720:rate=30 \
  -f lavfi -i sine=frequency=1000:sample_rate=48000 \
  -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
  -b:v 3500k -maxrate 4000k -bufsize 6000k -g 60 -r 30 \
  -c:a aac -b:a 128k -ar 48000 -ac 1 \
  -flvflags no_duration_filesize \
  -f flv "$YOUTUBE_RTMP_URL"
```

## Automated Film Analysis

The `analysis` package provides a small end-to-end pipeline that ingests a
full game video, performs lightweight play recognition and writes summary
artefacts. Run it with:

```bash
python -m analysis.pipeline --video path/to/game.mp4 --team WHITE --playbook playbooks/mca_5th_playbook.json --out output/ --generate-report
```

The command creates JSON lines files and, when `--generate-report` is used, a
coach report under `output/reports/`.

The coach summary report includes per-play tables and player grades. A
sample output is generated during tests under `tests/data`.

### One-click end-to-end analysis

Run the entire processing pipeline and summary generation with a single command:

```bash
python3 scripts/one_click_analyze.py \
  --video video/manual_uploads/IMG_4129.MP4 \
  --team WHITE \
  --opponent "Victory Christian" \
  --date 2025-08-08
```

## Google Drive sync & storage cleanup

One-time setup:

1. Create a Google Cloud service account with the Drive API enabled.
2. Share the target Drive folders (`GDRIVE_FOLDER_RAW`, `GDRIVE_FOLDER_ANALYZED`) with the service account email.
3. Save the JSON key locally and set `GDRIVE_CREDENTIALS_JSON` in your `.env`.

Uploads are verified via MD5/SHA1 before any local file is removed. Each upload is
recorded in `output/manifest.jsonl` with checksums and the Drive file ID. Use
`--verify-drive` and `--purge-now` to require verified uploads before deletion.

Example commands:

```bash
cd ~/mca-gameday-camera
OUT=output/IMG_4129_$(date +%Y%m%d_%H%M)
mkdir -p "$OUT"

PYTHONPATH=. python3 -m analysis.pipeline \
  --video video/manual_uploads/IMG_4129.MP4 \
  --team WHITE \
  --playbook playbooks/mca_5th_playbook.json \
  --out "$OUT" \
  --sync-to-drive

# Run sync/cleanup anytime (requires verified Drive uploads)
python3 tools/sync_and_cleanup.py --verify-drive --purge-now

# Cloud-first flow: download new Drive files since Aug 1, 2025
python3 tools/sync_and_cleanup.py --cloud-first --since 2025-08-01

# Download a specific Drive file id
python3 tools/sync_and_cleanup.py --cloud-first --id <drive_id>
```

To automate syncing every hour via `systemd`:

```bash
sudo cp systemd/mca-sync.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mca-sync.timer
systemctl list-timers | grep mca-sync
```

Or add a cron entry:

```
15 * * * * cd ~/mca-gameday-camera && /usr/bin/env -S bash -lc 'source .env && python3 tools/sync_and_cleanup.py --verify-drive --purge-now' >> ~/mca-gameday-camera/logs/sync.log 2>&1
```

## Playbook

Playbooks may be authored in a legacy flat list format or using the
new `split_sections` schema.  The latter separates offense and defense
sections:

```json
{
  "schema": "split_sections",
  "offense": {"plays": [{"name": "Rit Dive", "formation": "Rit"}]},
  "defense": {
    "positions": [{"name": "DT1", "gap": "A"}],
    "calls": [{"cue": "RUN", "trigger": "downfield blocking"}]
  }
}
```

When using `split_sections` the `defense.positions` array is required and
the pipeline will raise an error if it is missing.  Defensive grading weights
can be customised by editing
`analysis/configs/grading_weights_defense.yaml`; defaults are used when the
file is absent.

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

The default settings use the Jetson `h264_v4l2m2m` hardware encoder at
640x480 and 30fps with a bitrate around **2.5&nbsp;Mbps**.
Output is written with `tee` so a local MP4 recording is saved
alongside the live RTMP stream.

If `GDRIVE_FOLDER_ID` is set, the MP4 and a matching
`*_play_log.csv` are uploaded to that Drive folder after streaming
finishes. Set `GDRIVE_USE_GAME_FOLDER=1` to place both files in a
dedicated subfolder named after the game timestamp.

Additional options:

```bash
python stream_to_youtube.py --output-size 640x480 --debug
```

### Audio tuning

The mic input can be adjusted on the fly via environment variables. Defaults
favor sideline speech but can be tweaked for different environments:

```bash
# Stronger leveling for a noisy crowd
export AUDIO_MODE=crowd
export AUDIO_GAIN_DB=10

# Cut more wind/rumble
export AUDIO_HIGHPASS=150

# If things sound over-compressed
export AUDIO_GAIN_DB=6  # or set AUDIO_MODE=off
```
```bash
# Encoder preferences (comma-separated)
PREFERRED_ENCODERS=h264_v4l2m2m,libx264

# Force software (emergency switch)
USE_SW_ENC=0     # set to 1 to force libx264

# Bitrate tuning
VIDEO_BITRATE=3500k
VIDEO_MAXRATE=4000k
VIDEO_BUFSIZE=6000k

# Local recording of stream
RECORD_MP4=1
```

### Audio device quick start

List devices:

```bash
PYTHONPATH=. python3 -m tools.list_audio
```

Pulse explicit mic (replace with your exact pactl name from your log):

```bash
MIC_PULSE_NAME="alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_17477F5D-00.mono-fallback" \
  ./gameday --audio-source pulse
```

Allow-silent bypass (not recommended):

```bash
ALLOW_SILENT_STREAM=true ./gameday
```


## Requirements

- Python 3
- [OpenCV](https://opencv.org/)
- [ffmpeg](https://ffmpeg.org/)

Install FFmpeg on Jetson with:

```bash
sudo apt-get update && sudo apt-get install ffmpeg
```

For best performance on Jetson devices, the scripts use the Jetson
hardware encoder `h264_v4l2m2m` by default and automatically fall back
to `libx264` when no hardware encoder is available.

This repository contains simple utilities for analyzing football plays.

## Modules

- `play_classifier.py` – includes the `PlayClassifier` class for touchdown detection
  and a `classify_play` function to label short clips using a pretrained video model.
  Run `python play_classifier.py --folder clips/ --output predictions.json` to classify
  a directory of clips.
- `record_video.py` – records 640x480 video from /dev/video0 to output.mp4
- `highlight_recorder.py` – automatically captures 10-second clips when motion is detected
- `play_recognizer.py` – identifies plays based on formations in `playbooks/mca_5th_playbook.json` and writes results to `play_log.json`.
- `practice_trainer.py` – analyzes labeled practice clips and stores motion
  statistics in `training_set.json` for use by `play_recognizer.py`.
```bash
python play_recognizer.py path/to/game.mp4 --playbook playbooks/mca_5th_playbook.json --output play_log.csv
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

## MCA Film Analysis Pipeline

This repository now includes a minimal `mca_film` package that sketches an end-to-end
analysis workflow for the Metro Christian Academy 5th grade team. The pipeline
mirrors the coaching spec and is designed to be extended with real computer
vision models.

### Usage

```bash
python -m mca_film.cli analyze --video data/scrimmage.mp4 --side offense
python -m mca_film.cli export --report coaches --players p1 p2 --highlights
```

The first command runs the analysis and stores per-play JSON under `out/json/`.
The second command exports a coach summary CSV, per-player cutups and a simple
highlights reel placeholder under `out/`.

### Grading Heuristics

Current grading uses a neutral baseline of `2.0` for every player.  Future work
should expand this to measure contain responsibilities, gap fits and route
discipline as outlined in the coaching rubric:

- Defensive ends and edges are checked for outside contain.
- Tackles are evaluated on A/B gap integrity.
- Linebackers earn bonuses for visible read steps.
- Secondary players are graded on keeping the top of the coverage.

These rules are configurable via `config/settings.yaml`.


## Jetson quickstart

```bash
cd ~/mca-gameday-camera
OUT=output/scrimmage_$(date +%Y%m%d_%H%M)
mkdir -p "$OUT"

python3 -m analysis.pipeline \
  --video video/manual_uploads/IMG_4129.MP4 \
  --team WHITE \
  --playbook playbooks/mca_5th_playbook.json \
  --out "$OUT" \
  --player-ids config/player_visual_ids.yaml \
  --generate-report \
  --clip-corrections --clip-wins --clip-highlights
```

### Detector probe
Ensure imports work outside `-m`:

```
PYTHONPATH=. python3 tools/probe_detector.py
```

Tune at runtime:

```
MCA_DET_CONF=0.20 MCA_DET_NMS=0.55 MCA_DET_WEIGHTS=models/player_detector/best.onnx make run-pipeline
```
Use `--force-cpu` if GPU runtime is misconfigured.

## Development

For quick lint checks, install Pyflakes:

```
pip install pyflakes
```

# Add to ~/.bashrc
alias gameday_rode='USE_LOUDNORM=true EXTRA_GAIN_DB=4 ./gameday --audio-source pulse'


## Gameday
cp .env.example .env
# put your STREAM_KEY

./preflight
./gameday    # launches tmux with ffmpeg + log tail

# If you need to detach:
Ctrl-b d   # (tmux detach)
tmux attach -t gameday

### Gameday quick start

```bash
# Optional: override via env
export YOUTUBE_RTMP_URL='rtmps://a.rtmps.youtube.com/live2/<your-key>'
export VIDEO_DEV=/dev/video0
export PULSE_DEV='alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_17477F5D-00.mono-fallback'
# If MJPEG is flaky:
# export INPUT_FORMAT=yuyv422

# Enable debug echoing of the ffmpeg command
export DEBUG=1

# Launch
./gameday
```

Expected log lines:

- Using Pulse source: alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_...
- No appearance of min or max comp settings.
- A single `-af` string that ends with `aresample=async=1:first_pts=0`.

The launcher:

Auto-picks the RØDE if present.

Uses a robust audio chain (without min or max comp tweaks).

Adds large input queues and stable timestamps to reduce V4L2/DTS hiccups.

Retries on transient RTMPS/TLS failures.


## Acceptance tests

```bash
# B) Test pattern to YouTube (use real key; hold for 60–90s)
export YOUTUBE_RTMP_URL='rtmps://a.rtmps.youtube.com/live2/xxxx-xxxx-xxxx-xxxx-xxxx'
./gameday-testsrc

# C) Live devices to YouTube (use real key)
export VIDEO_DEV=/dev/video0
export PULSE_DEV='alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_17477F5D-00.mono-fallback'
export VIDEO_SIZE=1280x720
export FPS=30
export VIDEO_BITRATE=3500k
export VIDEO_MAXRATE=4000k
export VIDEO_BUFSIZE=6000k
./gameday

# D) RTMP vs RTMPS
export YOUTUBE_RTMP_URL='rtmps://b.rtmps.youtube.com/live2/xxxx-xxxx-xxxx-xxxx-xxxx'
./gameday-testsrc
```

## Notes for operators

- **Keys:** Reusable keys and scheduled events use different keys. Always paste the exact key for the stream you are monitoring in Live Control Room.
- **Latency:** Keep `-g` ≈ 2s (e.g., 60 for 30 fps). Set Low/Normal latency in YouTube unless chasing Ultra‑Low.
- **Bitrate tips:** 720p30 works well at 3500k target / 4000k max.
- **Devices busy:** If `/dev/video0` is busy:
  - `pkill -9 -f 'cheese|guvcview|nvarguscamerasrc|nvargus-daemon|ffmpeg.*video4linux2|gst-launch'`
  - `sudo systemctl stop nvargus-daemon` (harmless if unused).
- **Mic mono vs stereo:** We send mono with `-ac 1`. If you truly want stereo, change to `-ac 2`.

## Gameday Quick Start

```bash
# 0) Install deps
sudo apt-get update && sudo apt-get install -y ffmpeg jq ca-certificates
sudo timedatectl set-ntp true
sudo systemctl restart systemd-timesyncd || true
sudo update-ca-certificates

# 1) Pick devices
export VIDEO_DEV=/dev/video0
export PULSE_DEV='alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_XXXXXXXX-00.mono-fallback'

# 2) Put your YouTube key (no spaces, no angle brackets)
export YOUTUBE_RTMP_URL='rtmps://a.rtmps.youtube.com/live2/xxxx-xxxx-xxxx-xxxx-xxxx'

# 3) Test YouTube ingest with synthetic pattern
TESTSRC=1 ./gameday   # stop with Ctrl+C, confirm "Excellent" in YouTube Studio

# 4) Go live from camera + mic
./gameday
```

### Tips

If port 1935 is blocked, always use rtmps://...:443 (default here).

If you see “device busy”, stop any apps using /dev/video0 and nvargus-daemon if CSI sensors are present:

```
pkill -9 -f 'cheese|guvcview|nvargus|video4linux2|opencv|gst-launch|ffmpeg'
sudo systemctl stop nvargus-daemon || true
```

If Studio shows “low bitrate”, increase -b:v/-maxrate/-bufsize in gameday.
