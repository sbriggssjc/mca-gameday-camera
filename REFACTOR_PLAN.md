# Refactor Plan

## Shared Utilities
- Create `analysis/core/` package to hold shared helpers.
- Move duplicate file/path helpers into `analysis/core/io_utils.py`.
- Consolidate FFmpeg wrappers into `analysis/core/media_utils.py`.
- Centralize OpenCV/vision helpers in `analysis/core/vision_utils.py`.
- Gather model loading and inference helpers into `analysis/core/ml_utils.py`.
- Add timecode and SMPTE math to `analysis/core/timecode.py`.
- Provide structured logging helpers in `analysis/core/log_utils.py`.
- Add concurrency helpers in `analysis/core/concurrency.py`.
- Introduce unified configuration loader in `analysis/core/config.py`.

## Deduplication Targets
- Replace calls to `ffmpeg_utils.py`, `analysis/clipper.py`, and script-level FFmpeg snippets with `media_utils`.
- Replace scattered `os`/`pathlib` helpers with `io_utils` (keep thin wrappers for compatibility).
- Migrate repeated logging setup to `log_utils`.
- Standardize model loading across `analysis/*classifier*.py`, `ai_detector.py`, etc. via `ml_utils`.
- Use unified config loader for `config.py`, `analysis/config.py`, `env_loader.py`, and script flags.

## Output & Folder Structure
- Standardize outputs under `output/{DATE_OR_JOB}/` with subfolders for clips, frames, reports, artefacts, and logs.
- Ensure reruns are idempotent and reuse caches unless `--force` is passed.

## Testing & Tooling
- Introduce `pytest` with golden CLI tests and unit tests for core utils.
- Add `pre-commit` with `ruff`, `black`, and `mypy` in gradual mode.

## Migration
- Document moved/renamed functions in `MIGRATION.md` and provide backward-compatible wrappers with deprecation warnings.

