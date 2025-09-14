import os
import pathlib
import types
import subprocess
import shutil
import pytest
from analysis import pipeline

def test_clip_symlink(tmp_path, monkeypatch):
    pb = pathlib.Path("playbooks/mca_5th_playbook.json")
    monkeypatch.setattr(pipeline, "segment_video", lambda *a, **k: [{"id": "PLAY_001", "t0": 0.0, "t1": 5.0}])

    def fake_classify(segments, playbook, team, *, play_ckpt=None, play_labels=None, formation_ckpt=None, formation_labels=None, weak_threshold=0.35, smooth_frames=4):
        return [{
            "play_id": segments[0]["id"],
            "clf_top1": "Rit Jet Sweep",
            "clf_top1_conf": 0.9,
            "clf_top3": [("Rit Jet Sweep", 0.9)],
            "play_family": "Rit Jet Sweep",
        }]

    monkeypatch.setattr(pipeline, "classify_plays", fake_classify)
    monkeypatch.setattr(pipeline, "_ffmpeg", lambda *a, **k: types.SimpleNamespace(returncode=0))

    video = tmp_path / "dummy.mp4"
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    subprocess.run([
        "ffmpeg",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=160x120:d=1",
        str(video),
        "-y",
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    run_dir = pipeline.run_pipeline(
        video=str(video),
        team="WHITE",
        playbook_path=str(pb),
        out_dir=str(tmp_path),
        generate_clips=True,
        require_classifier=False,
    )

    run_dir = pathlib.Path(run_dir)
    link = run_dir / "clips" / "PLAY_001__Rit_Jet_Sweep"
    target = run_dir / "clips" / "PLAY_001"
    assert link.is_symlink()
    assert link.resolve() == target.resolve()
