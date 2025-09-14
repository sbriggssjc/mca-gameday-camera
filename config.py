"""Deprecated configuration helpers.

Use :mod:`analysis.core.config` instead.
"""
from __future__ import annotations

import argparse
import warnings
from typing import Any, Dict

from analysis.core.config import load_config as _load_config, get_cfg as _get_cfg


def load_config(path: str | None, args: argparse.Namespace) -> Dict[str, Any]:
    """Compatibility wrapper for the new configuration loader."""
    warnings.warn(
        "Deprecated, use analysis.core.config.load_config",
        DeprecationWarning,
        stacklevel=2,
    )
    cli = vars(args) if args else {}
    return _load_config(cli)


def get_config() -> Dict[str, Any]:
    warnings.warn(
        "Deprecated, use analysis.core.config.get_cfg",
        DeprecationWarning,
        stacklevel=2,
    )
    return _get_cfg()
