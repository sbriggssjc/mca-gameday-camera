import json, os, pathlib
import torch
from analysis import pipeline


def test_pipeline_label_mismatch(tmp_path, monkeypatch):
    playbook = {"plays": [{"name": "Rit Sweep", "formation": "Rit"}]}
    pb_path = tmp_path / "playbook.json"
    pb_path.write_text(json.dumps(playbook))

    model_ckpt = tmp_path / "model.pt"
    torch.save({"label_map": {"Rit Sweep": 0, "Foo": 1}}, model_ckpt)
    monkeypatch.setenv("PLAY_CLASSIFIER_MODEL", str(model_ckpt))

    run_dir = pipeline.run_pipeline(
        video="dummy.mp4",
        team="WHITE",
        playbook_path=str(pb_path),
        out_dir=str(tmp_path / "out"),
        min_play_gap=1.5,
        min_play_length=3.0,
        generate_report=True,
        generate_clips=False,
    )

    run_dir = pathlib.Path(run_dir)
    warn = (run_dir / "report" / "warnings.txt").read_text()
    assert "Foo" in warn
    html = (run_dir / "report" / "index.html").read_text()
    assert "0 segments detected" in html
    assert "Warnings" in html
    log_txt = (run_dir / "pipeline.log").read_text()
    assert "loading ckpt" in log_txt
    assert "labels: 2" in log_txt
