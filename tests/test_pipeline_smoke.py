import csv, json, os, pathlib, torch, subprocess, shutil, pytest
from analysis import pipeline


def test_pipeline_smoke(tmp_path):
    playbook = {"plays": [{"name": "Rit Sweep", "formation": "Rit", "motion": "sweep"}]}
    playbook_path = tmp_path / "playbook.json"
    playbook_path.write_text(json.dumps(playbook))

    video = tmp_path / "dummy.mp4"
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:d=1",
            str(video),
            "-y",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out_dir = tmp_path / "out"

    ckpt = tmp_path / "model.pt"
    torch.save({"label_map": {"Rit Sweep": 0}}, ckpt)
    os.environ["PLAY_CLASSIFIER_MODEL"] = str(ckpt)
    labels = tmp_path / "labels.txt"
    labels.write_text("Rit Sweep\n")
    f_ckpt = tmp_path / "formation.pt"
    torch.save({}, f_ckpt)
    f_labels = tmp_path / "formation_labels.txt"
    f_labels.write_text("Rit\n")

    run_dir = pipeline.run_pipeline(
        video=str(video),
        team="WHITE",
        playbook_path=str(playbook_path),
        out_dir=str(out_dir),
        play_labels=str(labels),
        formation_ckpt=str(f_ckpt),
        formation_labels=str(f_labels),
        min_play_gap=1.5,
        min_play_length=3.0,
        generate_report=True,
        generate_clips=False,
    )

    run_dir = pathlib.Path(run_dir)
    csv_path = run_dir / "plays_index.csv"
    assert csv_path.exists()
    assert (run_dir / "report.json").exists()
    index_path = run_dir / "report" / "index.html"
    assert index_path.exists()
    html = index_path.read_text()
    assert "0 segments detected" in html
    warnings_path = run_dir / "report" / "warnings.txt"
    assert warnings_path.exists()
    warn_txt = warnings_path.read_text()
    assert "torch:" in warn_txt
    assert "Warnings" in html

    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
    assert "smoothing_applied" in header
