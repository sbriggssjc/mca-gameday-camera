@echo off

REM Launch highlight_recorder.py on remote Jetson Nano
REM Usage: double-click or run from command prompt

set USER=scott
set HOST=192.168.6.35

ssh %USER%@%HOST% "python3 ~/mca-gameday-camera/highlight_recorder.py"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Failed to connect or run highlight_recorder.py on %HOST%.
    echo Ensure the Jetson Nano is reachable and SSH credentials are correct.
)

pause
