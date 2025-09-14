"""Deprecated configuration helpers.

Use :mod:`analysis.core.config` instead.
"""
from __future__ import annotations

import argparse
import warnings
from typing import Any, Dict

from analysis.core.config import load_config as _load_config, get_cfg as _get_cfg

_WARNED = False


def _warn() -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(
            "config module is deprecated; use analysis.core.config",
            DeprecationWarning,
            stacklevel=2,
        )
        _WARNED = True


def load_config(path: str | None, args: argparse.Namespace) -> Dict[str, Any]:
    """Compatibility wrapper for the new configuration loader."""
    _warn()
    cli = vars(args) if args else {}
    return _load_config(cli)


def get_config() -> Dict[str, Any]:
    _warn()
    return _get_cfg()
