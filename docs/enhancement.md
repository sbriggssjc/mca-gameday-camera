# Video Enhancement Tools

This repository provides utilities to stabilise, denoise and upscale game film.

## Batch Enhancement

Use `scripts/enhance_batch.sh` to process a directory of clips.

```bash
./scripts/enhance_batch.sh INPUT_DIR OUTPUT_DIR [ZOOM] [BITRATE]
```

- **ZOOM** – optional center crop (default `0.95`). Values `0.5-1.0` keep the field in view.
- **BITRATE** – target video bitrate (default `10M`).
- Hardware encoding uses `h264_v4l2m2m` when available and falls back to `libx264`.
- Stabilisation via `vid.stab` runs automatically when the filter is present.

Each enhanced clip is written as `*_enh1080p.mp4` in the output directory.

## Coaches Cut

`scripts/make_coaches_cut.sh` enhances today's clips and builds a single
coaches-cut video.

```bash
./scripts/make_coaches_cut.sh [DATE] [OUTDIR] [MODE]
```

- **DATE** – `YYYYMMDD` (defaults to today).
- **OUTDIR** – destination for the final MP4.
- **MODE** – `clips`, `raw` or `both` (default `clips`).

Zoom is chosen per clip using play metadata when available
(`run`→0.90, `pass`→0.98, `punt/kick`→1.00, else 0.95).

## Troubleshooting

- **Missing vid.stab filters** – stabilisation is skipped automatically.
  Install `ffmpeg` with vid.stab support if required.
- **Hardware encoder busy** – the scripts retry with `libx264 -preset veryfast -crf 18`.
- **Truncated files** – files still being written are skipped using a size check and
  `ffprobe` validation.
- **Bitrate** – 10–12M works well for 1080p output; increase for higher quality.
