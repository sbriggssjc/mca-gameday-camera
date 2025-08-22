@echo off
REM Launch livestream and play count tracker

start cmd /k "python stream_to_youtube.py"
start cmd /k "python play_count_tracker.py"
