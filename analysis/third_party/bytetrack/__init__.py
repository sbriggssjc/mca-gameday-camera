"""Lightweight ByteTrack implementation used for tests.

This module intentionally provides only a very small subset of the real
ByteTrack algorithm.  It implements IOU based association with a simple
velocity smoothing.  The goal of the vendor copy is to avoid pulling in a
heavy dependency while keeping the public API similar to the real tracker.

The tracker is good enough for unit tests and for running on the small
example clips that accompany the repository.  It is **not** intended to be
production quality.
"""

from .tracker import ByteTracker

__all__ = ["ByteTracker"]

