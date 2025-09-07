# Clip Enhancement

Add-on filters to stabilize, denoise and upscale clips. Works as a batch
process or as an optional post-step in the analysis pipeline.

## Setup
```bash
chmod +x scripts/ai_upscale.sh scripts/stabilize.sh scripts/enhance_batch.sh scripts/install_realesrgan_ncnn.sh
```

## Batch existing clips
```bash
./scripts/enhance_batch.sh "output/coach_cut_20250906/clips" "output/coach_cut_20250906/enhanced_ai" 18 10M --ai --scale 2 --engine realesrgan --stabilize
```
Fast (no stabilization): omit `--stabilize`. If `realesrgan-ncnn-vulkan` is not
installed the script falls back to FFmpeg upscaling.

## Install Real-ESRGAN (optional)
```bash
./scripts/install_realesrgan_ncnn.sh
```
Follow the printed steps; if you skip, the pipeline uses FFmpeg fallback.

## Troubleshooting
- "No videos in ..." → check input directory/glob.
- Missing `realesrgan-ncnn-vulkan` → AI upscaling falls back to FFmpeg.
- Missing `vidstabdetect`/`vidstabtransform` → stabilization step skipped.
- Use bitrate 10–12M; higher for fewer artifacts.
