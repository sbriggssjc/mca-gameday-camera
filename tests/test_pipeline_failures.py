import csv
import json
import pytest

from analysis import pipeline


def test_pipeline_no_run_dir_on_load_failure(tmp_path):
    out_dir = tmp_path / "out"
    playbook_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline(
            video="dummy.mp4",
            team="WHITE",
            playbook_path=str(playbook_path),
            out_dir=str(out_dir),
        )

    assert not (out_dir / "games").exists()


def test_pipeline_marks_failed_run(tmp_path, monkeypatch):
    playbook = {"plays": [{"name": "Rit Sweep", "formation": "Rit"}]}
    pb_path = tmp_path / "playbook.json"
    pb_path.write_text(json.dumps(playbook))

    out_dir = tmp_path / "out"

    class BoomWriter(csv.DictWriter):
        def writeheader(self):  # type: ignore[override]
            raise RuntimeError("boom")

    monkeypatch.setattr(
        pipeline.csv, "DictWriter", lambda f, fieldnames: BoomWriter(f, fieldnames)
    )

    with pytest.raises(RuntimeError):
        pipeline.run_pipeline(
            video="dummy.mp4",
            team="WHITE",
            playbook_path=str(pb_path),
            out_dir=str(out_dir),
        )

    games_dir = out_dir / "games"
    run_dirs = list(games_dir.glob("*"))
    assert run_dirs, "run dir should exist"
    run_dir = run_dirs[0]
    assert (run_dir / "RUN_FAILED.txt").exists()

