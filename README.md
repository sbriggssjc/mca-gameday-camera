# mca-gameday-camera

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

