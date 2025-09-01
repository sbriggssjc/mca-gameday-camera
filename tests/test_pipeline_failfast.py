import json, pathlib, pytest, sys, types

sys.modules.setdefault("numpy", types.SimpleNamespace())
sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        save=lambda *a, **k: None,
        load=lambda *a, **k: {},
        __version__="0.0",
        cuda=types.SimpleNamespace(
            is_available=lambda: False, get_device_name=lambda _i: "cpu"
        ),
    ),
)

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
    assert not run_dirs


def test_disabled_classifier_writes_warning(tmp_path, monkeypatch):
    pb = {"plays": []}
    pb_path = tmp_path / "playbook.json"
    pb_path.write_text(json.dumps(pb))

    out_dir = tmp_path / "out"

    monkeypatch.setattr(pipeline, "segment_video", lambda *a, **k: [])

    run_dir = pipeline.run_pipeline(
        video="dummy.mp4",
        team="WHITE",
        playbook_path=str(pb_path),
        out_dir=str(out_dir),
        require_classifier=False,
    )
    warn_path = pathlib.Path(run_dir) / "report" / "warnings.txt"
    assert warn_path.exists(), "warnings file missing"
    assert "classifier disabled by flag" in warn_path.read_text()
