# mca-gameday-camera

This repository contains simple utilities for analyzing football plays.

## Modules

- `play_classifier.py` – provides the `PlayClassifier` class which wraps a
  YOLOv5 model to detect touchdown-like movements or fast exits of a
  player from the frame.
