# mca-gameday-camera

This repository contains utilities for tracking play participation during a game.

## play_count_tracker.py

`play_count_tracker.py` is a simple command line tool for recording which players were on the field for each play. After each play, type the jersey numbers (1-17) separated by spaces. Enter `q` to finish. A log is written to `play_log.csv` and the final play counts are printed with alerts for any player that participated in fewer than seven plays.

### Usage

```bash
python play_count_tracker.py
```
