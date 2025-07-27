#!/bin/sh
# Upload all MP4 files in the video directory to Google Drive
DIR="$(dirname "$0")"
cd "$DIR" || exit 1
FILES=$(ls video/*.mp4 2>/dev/null)
if [ -n "$FILES" ]; then
    python3 upload_to_drive.py $FILES
fi
