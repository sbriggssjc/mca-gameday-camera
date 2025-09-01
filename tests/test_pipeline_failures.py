import csv, json, os, pathlib, pytest, sys, types

def _dummy_save(obj, path):
    import pickle

    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def _dummy_load(path):
    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)


sys.modules["numpy"] = types.SimpleNamespace()
sys.modules["torch"] = types.SimpleNamespace(
    save=_dummy_save,
    load=_dummy_load,
    __version__="0.0",
    cuda=types.SimpleNamespace(
        is_available=lambda: False, get_device_name=lambda _i: "cpu"
    ),
)

classifiers_stub = types.ModuleType("analysis.classifiers")
classifiers_stub.load_models = lambda args: types.SimpleNamespace()
classifiers_stub._load_ckpt = lambda path: {}
classifiers_stub._load_labels = lambda path: []
classifiers_stub.log = types.SimpleNamespace(info=lambda *a, **k: None)
sys.modules["analysis.classifiers"] = classifiers_stub

import torch  # type: ignore
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

    monkeypatch.setattr(pipeline, "segment_video", lambda *a, **k: [{"t0": 0.0, "t1": 1.0}])

    monkeypatch.setattr(
        pipeline,
        "classify_plays",
        lambda segments, *args, **kwargs: [
            {
                "play_id": seg.get("id", f"PLAY_{i+1:03d}"),
                "formation": "",
                "formation_confidence": 0.0,
                "play_family": "",
                "playcall_confidence": 0.0,
                "clf_top1": "",
                "clf_top1_conf": 0.0,
                "candidates": [],
            }
            for i, seg in enumerate(segments)
        ],
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

