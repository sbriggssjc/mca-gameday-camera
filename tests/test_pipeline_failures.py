import csv, json, os, pathlib, pytest, torch

from analysis import pipeline


def test_pipeline_no_run_dir_on_load_failure(tmp_path):
    out_dir = tmp_path / "out"
    playbook_path = tmp_path / "missing.json"
    play_ckpt = tmp_path / "model.pt"
    torch.save({}, play_ckpt)
    labels = tmp_path / "labels.txt"
    labels.write_text("Rit\n")
    f_ckpt = tmp_path / "formation.pt"
    torch.save({}, f_ckpt)
    f_labels = tmp_path / "formation_labels.txt"
    f_labels.write_text("Rit\n")

    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline(
            video="dummy.mp4",
            team="WHITE",
            playbook_path=str(playbook_path),
            out_dir=str(out_dir),
            play_ckpt=str(play_ckpt),
            play_labels=str(labels),
            formation_ckpt=str(f_ckpt),
            formation_labels=str(f_labels),
        )
    games_dir = out_dir / "games"
    assert not games_dir.exists()


def test_pipeline_marks_failed_run(tmp_path, monkeypatch):
    playbook = {"plays": [{"name": "Rit Sweep", "formation": "Rit"}]}
    pb_path = tmp_path / "playbook.json"
    pb_path.write_text(json.dumps(playbook))

    out_dir = tmp_path / "out"

    # Provide a dummy model so label loading succeeds
    ckpt = tmp_path / "model.pt"
    torch.save({"label_map": {"Rit Sweep": 0}}, ckpt)
    monkeypatch.setenv("PLAY_CLASSIFIER_MODEL", str(ckpt))
    labels = tmp_path / "labels.txt"
    labels.write_text("Rit Sweep\n")
    f_ckpt = tmp_path / "formation.pt"
    torch.save({}, f_ckpt)
    f_labels = tmp_path / "formation_labels.txt"
    f_labels.write_text("Rit\n")

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
            play_labels=str(labels),
            formation_ckpt=str(f_ckpt),
            formation_labels=str(f_labels),
        )

    games_dir = out_dir / "games"
    run_dirs = list(games_dir.glob("*"))
    assert run_dirs, "run dir should exist"
    run_dir = run_dirs[0]
    assert (run_dir / "RUN_FAILED.txt").exists()

