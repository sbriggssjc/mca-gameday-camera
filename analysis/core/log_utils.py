"""Project-wide logging helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def init_logger(name: str, json: bool = False, level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    """Initialise and return a logger with optional JSON and file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    formatter: logging.Formatter
    if json:
        try:
            import json_log_formatter  # type: ignore

            formatter = json_log_formatter.JSONFormatter()
        except Exception:  # pragma: no cover - fallback
            formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.setLevel(level)
    return logger


__all__ = ["init_logger"]
