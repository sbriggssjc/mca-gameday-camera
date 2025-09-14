"""Video enhancement helpers used by the aerial replay pipeline."""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Iterable, List


def is_tool_on_path(name: str) -> bool:
    """Return ``True`` when ``name`` resolves via ``PATH``."""
    return shutil.which(name) is not None


def _run_ffmpeg(args: List[str]) -> int:
    return subprocess.run(["ffmpeg", "-y", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode


def stabilize_ffmpeg(inp: str, out: str) -> bool:
    """Apply a basic stabilization using ffmpeg's vidstab filters."""

    if shutil.which("ffmpeg") is None:
        return False
    trf = out + ".trf"
    ret = _run_ffmpeg(["-i", inp, "-vf", f"vidstabdetect=shakiness=5:accuracy=15:result={trf}", "-f", "null", "-"])
    if ret != 0 or not os.path.exists(trf):
        return False
    ret = _run_ffmpeg(["-i", inp, "-vf", f"vidstabtransform=input={trf}", out])
    if ret != 0:
        return False
    return True



def superres_realesrgan(inp: str, out: str, scale: int = 2) -> bool:
    """Invoke ``realesrgan-ncnn-vulkan`` if available."""

    if not is_tool_on_path("realesrgan-ncnn-vulkan"):
        return False
    ret = subprocess.run(
        ["realesrgan-ncnn-vulkan", "-i", inp, "-o", out, f"-s{scale}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return ret.returncode == 0


def deblur_ffmpeg(inp: str, out: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    return _run_ffmpeg(["-i", inp, "-vf", "unsharp=lx=7:ly=7:la=0.9,hqdn3d=1:1:6:6", out]) == 0


def color_tune_ffmpeg(inp: str, out: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    return _run_ffmpeg(["-i", inp, "-vf", "eq=contrast=1.1:saturation=1.1:gamma=1.0", out]) == 0


PRESETS = {
    "fast": ["stabilize", "color_tune"],
    "max": ["stabilize", "superres", "deblur", "color_tune"],
}


STEP_FUNCS = {
    "stabilize": stabilize_ffmpeg,
    "superres": superres_realesrgan,
    "deblur": deblur_ffmpeg,
    "color_tune": color_tune_ffmpeg,
}


def enhance_pipeline(inp: str, out: str, steps: Iterable[str]) -> str:
    """Run enhancement ``steps`` sequentially writing result to ``out``.

    Missing steps are ignored gracefully; if none succeed the input file is
    simply copied to ``out``.  The path to the final processed file is returned
    which may be ``out`` or the input when processing fails.
    """

    current = inp
    tmp_out = out
    for step in steps:
        func = STEP_FUNCS.get(step)
        if not func:
            continue
        ok = func(current, tmp_out)
        if ok:
            current = tmp_out
    if current != out:
        shutil.copy(current, out)
    return out
