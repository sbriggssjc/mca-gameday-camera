# Repository Inventory

## Entry Points / CLIs
- Python scripts in repo root (e.g., `ai_detector.py`, `gameday_capture.py`, `run_gameday.py`, `generate_highlights.py`, etc.).
- Analysis pipeline (`analysis/pipeline.py`) and related modules under `analysis/`.
- Various helper scripts in `scripts/` directory (`run_game.sh`, `gameday_launcher.sh`, `preflight.py`, etc.).
- Windows batch files (`run_gameday_highlight.bat`, `start_gameday.bat`, etc.).

## Major Modules
- `analysis/` – football video analysis pipeline and utilities.
- `gameday/`, `gameday.bak*` – historical or backup pipeline versions.
- Root-level utilities: `ffmpeg_utils.py`, `env_loader.py`, `config.py`, etc.

## Data Folders
- `models/`, `models_bundle/` – ML model weights.
- `recordings/`, `outputs/`, `output/` – raw recordings and processed outputs.
- `playbooks/`, `plays/`, `player_id/` – play definitions and datasets.
- `logs/` – run-time logs.
- `templates/`, `overlays/`, `configs/`, `hooks/` – configuration and assets.

## Reused Utilities
- Custom FFmpeg wrappers (`ffmpeg_utils.py`, `analysis/clipper.py`, `scripts/ffmpeg` helpers).
- File and path helpers (`analysis/io_utils.py`, `env_loader.py`, scattered `config.py` files).
- Logging utilities spread across scripts.
- Detection/classification helpers (`analysis/*classifier*.py`, `ai_detector.py`, `play_classifier.py`).
- Tracking/zoom utilities (`analysis/zoom.py`, `auto_tracker.py`, `analysis/autozoom.py`).
- Report generation (`analysis/report*.py`, `generate_coaching_report.py`).

## Duplicate / Near-Duplicate Functions
- Multiple FFmpeg invocation helpers across `ffmpeg_utils.py`, `analysis/clipper.py`, `scripts/*`.
- Repeated file/path utilities in `analysis/io_utils.py`, `env_loader.py`, `gdrive_utils.py`.
- Similar logging setup in many scripts.
- Model loading logic scattered in `analysis` and root scripts.
- Configuration parsing duplicated in `config.py`, `analysis/config.py`, and `env_loader.py`.

