import json
import os
import time
from pathlib import Path

import pytest

from tools import gdrive_sync, storage_cleanup
from tools.json_io import iter_jsonl_safe


class FakeDrive:
    def files(self):
        return self

    # upload_file_with_md5 is patched, so create/list are unused
    def get(self, fileId, fields=None):
        self._meta = {
            "id": fileId,
            "name": "sample.txt",
            "md5Checksum": self.expected_md5,
            "size": "5",
            "parents": ["parent"],
        }
        return self

    def execute(self):
        return self._meta


def test_upload_and_remove_verified(tmp_path, monkeypatch):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello")

    manifest = tmp_path / "manifest.jsonl"

    md5 = gdrive_sync.hash_file(file_path)

    fake_drive = FakeDrive()
    fake_drive.expected_md5 = md5

    monkeypatch.setattr(gdrive_sync, "_drive", lambda: fake_drive)

    def fake_upload(drive, local_path, parent_id):
        return {"id": "fake123", "md5Checksum": md5}

    monkeypatch.setattr(gdrive_sync, "upload_file_with_md5", fake_upload)

    gdrive_sync.upload_tree_with_manifest(tmp_path, "parent", manifest)

    lines = list(iter_jsonl_safe(manifest))
    rec = next(r for r in lines if r["local"].endswith("sample.txt"))
    assert rec["status"] == "verified"
    assert rec["drive_id"]

    # create two run directories
    out = tmp_path / "out"
    run_old = out / "run_old"
    run_new = out / "run_new"
    run_old.mkdir(parents=True)
    run_new.mkdir(parents=True)
    f_old = run_old / "a.txt"
    f_new = run_new / "b.txt"
    f_old.write_text("a")
    f_new.write_text("b")

    # mark both files as verified in manifest
    with manifest.open("a") as mf:
        for p in [f_old, f_new]:
            rec = {
                "ts": "now",
                "local": str(p),
                "bytes": p.stat().st_size,
                "md5": "0",
                "sha1": "0",
                "drive_id": "id",
                "drive_md5": "0",
                "parents": ["p"],
                "status": "verified",
            }
            mf.write(json.dumps(rec) + "\n")

    # make old run older than retain_days
    old_time = time.time() - 2 * 24 * 3600
    os.utime(run_old, (old_time, old_time))
    os.utime(f_old, (old_time, old_time))

    removed = storage_cleanup.remove_old_runs_verified(
        out, retain_latest=1, retain_days=0, manifest_path=manifest
    )

    assert run_old in removed
    assert not run_old.exists()
    assert run_new.exists()

