"""Minimal calibration UI placeholder."""

from __future__ import annotations


def launch_ui(tracklets, roster):  # pragma: no cover - UI is not tested
    """Pretend to launch a calibration interface.

    A real implementation might use Streamlit or Flask.  Here we simply print a
    message so the function can be invoked in scripts without failing.
    """

    print("Calibration UI requested for", len(tracklets), "tracklets")
    _ = roster
