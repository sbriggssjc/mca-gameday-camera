# Migration Notes

This file documents API moves and deprecations introduced during the refactor.

## Core utilities

Common helpers now live under ``analysis.core``:

| Old location                 | New module                        |
|-----------------------------|-----------------------------------|
| ``analysis.io_utils``       | ``analysis.core.io_utils``        |
| ``ffmpeg_utils.ffprobe_json`` | ``analysis.core.media_utils.ffprobe_json`` |
| ``ffmpeg_utils.ffmpeg_cut`` | ``analysis.core.media_utils.ffmpeg_cut`` |
| ``config.load_config``      | ``analysis.core.config.load_config`` |
| ``env_loader.load_env``     | ``analysis.core.config.load_config`` |

The old symbols remain as thin wrappers issuing ``DeprecationWarning`` and
will be removed after **2025-12-31**.

## Output paths

Results are now written under ``output/<job>/``. A compatibility
symlink ``outputs`` is created on first use.  Update scripts to use the
new path. The alias will be removed after **2025-12-31**.

