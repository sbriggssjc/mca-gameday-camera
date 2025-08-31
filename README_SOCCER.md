# Soccer Gameday Recorder

This repo includes a minimal soccer recording path. Use `./soccerday` to start
recording from the capture card and microphone.

## Testing checklist

### A) Discover devices
```bash
PYTHONPATH=. python3 -m tools.list_av
# Note the Rode mic Pulse name, e.g.:
#   alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_XXXX-00.mono-fallback
```

### B) 10-second smoke test
```bash
AUDIO_SRC="alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_XXXX-00.mono-fallback" \
SOC_TITLE="smoketest" SOC_SEG_MIN=1 \
./soccerday
# Let it run ~10s, Ctrl+C
# Verify: output/soccer/full/ has an mp4, logs/ has a .log, thumbs/ has jpgs
# (if >10s), meta/ has json
```

### C) Input format fallback
```bash
SOC_INPUT_FORMAT=yuyv422 ./soccerday
# If your capture card prefers yuyv422
```

### D) Proxy/on-the-fly assets
```bash
SOC_PROXY=1 SOC_TITLE="withproxy" ./soccerday
# Check output/soccer/proxy for smaller files
```

### E) Space guard
```bash
# Temporarily set require_free_gb to a huge value (e.g., 9999) and confirm that
# recorder refuses to start with a clear message.
```

### F) Real match capture
```bash
AUDIO_SRC="alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_XXXX-00.mono-fallback" \
SOC_TITLE="U11_vs_Glenpool_20250903" SOC_SEG_MIN=20 \
./soccerday
# Let it roll the entire match. It will create multiple 20-minute parts.
```

### G) Playback sanity
Confirm the last segment plays and has audio (quick scrub in VLC).
Check the log for dropped-source errors.
Verify thumbs exist at roughly every N seconds for the full span.
<<<<<<< HEAD
=======

## Filename options
Set `SOC_TIMESTAMPS=1` to name segments by timestamp. The default uses part counters like `soccer_part000.mp4`.

## MKV segments
Set `SOC_CONTAINER=mkv` to produce MKV files which can be probed while they are still open. When using the default MP4 container, avoid probing the newest segment until recording stops (the "moov atom not found" message is expected).
>>>>>>> 3fb8c6c8bd1feab7561579284c161798bd1142cb
