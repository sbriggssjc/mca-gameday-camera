"""Unified configuration loader.

Configuration is merged from ``config/defaults.yaml`` (if
present), optional ``.env`` files and CLI overrides.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

_CFG: Dict[str, Any] | None = None


def _load_defaults() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    path = Path("config/defaults.yaml")
    if path.exists() and yaml:
        cfg.update(yaml.safe_load(path.read_text()) or {})
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg.setdefault(k.strip(), v.strip())
    return cfg


def load_config(cli: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return merged configuration with CLI overrides applied."""
    global _CFG
    if _CFG is None:
        _CFG = _load_defaults()
    if cli:
        cfg_path = cli.get("config")
        if cfg_path and yaml and Path(cfg_path).exists():
            _CFG.update(yaml.safe_load(Path(cfg_path).read_text()) or {})
        _CFG.update({k: v for k, v in cli.items() if v is not None})
    return _CFG


def get_cfg() -> Dict[str, Any]:
    """Return cached configuration."""
    if _CFG is None:
        load_config()
    assert _CFG is not None
    return _CFG


__all__ = ["load_config", "get_cfg"]
