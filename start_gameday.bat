@echo off
REM One-click launcher for game day camera operations

set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

if not exist logs mkdir logs

REM Load environment variables from .env if present
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if not defined %%A set %%A=%%B
    )
)

if "%YOUTUBE_RTMP_URL%"=="" (
    echo Missing YOUTUBE_RTMP_URL in environment or .env
    pause
    exit /b 1
)

REM Check camera availability using Python and OpenCV
python - <<PY
import cv2, sys
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit(1)
cap.release()
PY
if errorlevel 1 (
    echo Camera not found. Connect the camera and try again.
    pause
    exit /b 1
)

REM Start livestream
start "Livestream" cmd /k python stream_to_youtube.py

REM Start local recording
start "Recording" cmd /k python record_video.py

REM Start highlight recorder
start "Highlights" cmd /k python highlight_recorder.py

REM Start play count tracker
start "Play Tracker" cmd /k python play_count_tracker.py

REM TODO: add battery and network status checks

pause
