import json, pathlib, pytest
from analysis import pipeline


def test_require_classifier_missing_ckpt(tmp_path):
    pb = {"plays": []}
    pb_path = tmp_path / "playbook.json"
    pb_path.write_text(json.dumps(pb))

    out_dir = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline(
            video="dummy.mp4",
            team="WHITE",
            playbook_path=str(pb_path),
            out_dir=str(out_dir),
            play_ckpt=str(tmp_path / "missing.pt"),
            generate_report=True,
            require_classifier=True,
        )
    run_dirs = list((out_dir / "games").glob("*"))
    assert run_dirs
    warn_path = run_dirs[0] / "report" / "warnings.txt"
    assert warn_path.exists()
    warn = warn_path.read_text()
    assert "missing required file" in warn
    csv_path = run_dirs[0] / "plays_index.csv"
    assert not csv_path.exists()
