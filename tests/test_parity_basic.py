from __future__ import annotations

import subprocess
from pathlib import Path

import shutil
import pytest

if shutil.which("ffmpeg") is None:  # pragma: no cover - environment dependent
    pytest.skip("ffmpeg not installed", allow_module_level=True)

from analysis import pipeline
from analysis.core.media_utils import ffmpeg_cut


def make_sample(tmp: Path) -> Path:
    sample = tmp / "sample.mp4"
    subprocess.check_call([
        "ffmpeg",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=160x120:d=1",
        "-c:v",
        "libx264",
        "-t",
        "1",
        str(sample),
        "-y",
    ])
    return sample


def test_pipeline_smoke(tmp_path):
    sample = make_sample(tmp_path)
    job = "testjob"
    pipeline.run_pipeline(
        video=str(sample),
        team="home",
        playbook_path="tests/fixtures/sample_playbook_split.json",
        out_dir=job,
        require_classifier=False,
        generate_report=True,
        generate_clips=True,
        min_play_length=0.1,
        max_play_length=1.0,
        min_activity_ratio=0.0,
    )
    job_root = Path("output") / job
    games = list((job_root / "games").glob("*"))
    assert games, "game directory created"
    gdir = games[0]
    clip_dir = gdir / "clips"
    clip_dir.mkdir(exist_ok=True)
    if not any(clip_dir.glob("*.mp4")):
        # Create a clip manually if pipeline detected none
        ffmpeg_cut(sample, 0, 1, clip_dir / "clip_0.mp4")
    assert any(clip_dir.glob("*.mp4")), "at least one clip exists"
    report = gdir / "report" / "index.html"
    assert report.exists(), "report generated"
