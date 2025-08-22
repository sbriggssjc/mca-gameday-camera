"""
Alias loader for the actual player detector implementation.

This module looks for another module under `analysis.detectors.*`
that exposes a `detect(...)` function (and optionally DEFAULT_CONF,
DEFAULT_NMS, EXPECTS_RGB, INPUT_SIZE, WEIGHTS_PATH) and then
re-exports its public symbols so existing imports keep working:

    from analysis.detectors import player_detector
    boxes, scores, classes = player_detector.detect(...)

If you know the exact file (e.g., yolo_player.py), you can replace
this module with:
    from .yolo_player import *
"""
import pkgutil, importlib
from types import ModuleType

PKG_PREFIX = __name__.rsplit('.', 1)[0]  # 'analysis.detectors'


def _load_impl() -> ModuleType:
    preferred = ["yolo_player", "yolo_detector", "players", "detector", "people_detector"]
    for base in preferred:
        try:
            return importlib.import_module(f"{PKG_PREFIX}.{base}")
        except Exception:
            pass

    # Fallback: discover any module with a detect() function
    import analysis.detectors as _pkg
    for mod in pkgutil.walk_packages(_pkg.__path__, prefix=_pkg.__name__ + "."):
        name_l = mod.name.lower()
        if name_l.endswith("__init__"):
            continue
        if ("player" in name_l or "detect" in name_l) and "player_detector" not in name_l:
            try:
                m = importlib.import_module(mod.name)
                if hasattr(m, "detect"):
                    return m
            except Exception:
                continue
    raise ImportError("No detector implementation with detect() found under analysis.detectors.*")


_impl = _load_impl()

# Re-export everything public from the impl module
for k in dir(_impl):
    if not k.startswith("_"):
        globals()[k] = getattr(_impl, k)
__all__ = [k for k in globals().keys() if not k.startswith("_")]

