from __future__ import annotations

"""Helpers for working with playbook files."""

# Expose the loader at package level so callers may simply import
# `load_playbook` from ``analysis.playbook`` without reaching into the
# ``loader`` module directly.
from .loader import load_playbook

__all__ = ["load_playbook"]

