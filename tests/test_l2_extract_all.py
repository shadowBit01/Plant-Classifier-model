"""Tests for scripts/extract_all.py (Lane L2)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from scripts import extract_all
from vanalens import config, contracts


def _make_valid_npz(path: Path) -> None:
    X = np.zeros((2, config.EMB_DIM), dtype=np.float32)
    X[:, 0] = 1.0
    X[:, config.BIOCLIP_DIM] = 1.0
    es = contracts.EmbeddingSet(ids=["a", "b"], X=X, y=np.array([0, 1], dtype=np.int64))
    es.save(path)


def test_is_up_to_date(tmp_path: Path):
    m = tmp_path / "ds_test.jsonl"
    e = tmp_path / "ds_test.npz"
    m.write_text('{"image_id":"a","path":"x.jpg","label":0}\n', encoding="utf-8")
    assert not extract_all.is_up_to_date(m, e)

    _make_valid_npz(e)
    now = time.time()
    os.utime(m, (now - 10, now - 10))
    os.utime(e, (now, now))
    assert extract_all.is_up_to_date(m, e)

    os.utime(m, (now + 10, now + 10))
    assert not extract_all.is_up_to_date(m, e)


def test_run_extraction_skips_up_to_date_and_runs_stale(tmp_path: Path, monkeypatch):
    m_dir = tmp_path / "manifests"
    e_dir = tmp_path / "emb"
    log_file = tmp_path / "extract_all.log"
    m_dir.mkdir()
    e_dir.mkdir()

    m1 = m_dir / "a_gallery.jsonl"
    m2 = m_dir / "b_test.jsonl"
    m1.write_text('{"image_id":"1","path":"1.jpg","label":0}\n', encoding="utf-8")
    m2.write_text('{"image_id":"2","path":"2.jpg","label":0}\n', encoding="utf-8")

    e1 = e_dir / "a_gallery.npz"
    _make_valid_npz(e1)
    now = time.time()
    os.utime(m1, (now - 10, now - 10))
    os.utime(e1, (now, now))

    calls: list[list[str]] = []

    class DummyProc:
        def __init__(self, cmd, **kwargs):
            calls.append(cmd)
            out_idx = cmd.index("--out") + 1
            _make_valid_npz(Path(cmd[out_idx]))
            self.stdout = iter(["extracted b_test\n"])

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", DummyProc)

    outs = extract_all.run_extraction(manifests_dir=m_dir, emb_dir=e_dir, log_file=log_file)
    assert len(outs) == 2
    assert len(calls) == 1
    assert calls[0][:3] == [sys.executable, "-m", "vanalens.encoders"]
    assert str(m2) in calls[0]
    log_text = log_file.read_text(encoding="utf-8")
    assert "Up-to-date, skipping: a_gallery.jsonl" in log_text
    assert "ALL DONE" in log_text.splitlines()[-1]


def test_detach_flags(tmp_path: Path, monkeypatch):
    captured: dict = {}

    class DummyDetachProc:
        pid = 42424

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    monkeypatch.setattr(subprocess, "Popen", DummyDetachProc)
    log_file = tmp_path / "detach.log"
    extract_all.main(["--detach", "--batch-size", "16", "--log-file", str(log_file)])

    assert "--detach" not in captured["cmd"]
    assert "--batch-size" in captured["cmd"]
    if os.name == "nt":
        expected = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        assert captured["kwargs"]["creationflags"] == expected
    else:
        assert captured["kwargs"]["start_new_session"] is True


def test_encoders_cli_tta_and_gallery_aug(tmp_path: Path, monkeypatch):
    from PIL import Image
    from vanalens import encoders

    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    bad = tmp_path / "bad.jpg"
    Image.new("RGB", (64, 48), color=(20, 180, 40)).save(img1)
    Image.new("RGB", (64, 48), color=(180, 40, 20)).save(img2)
    bad.write_bytes(b"not a valid image")

    manifest = tmp_path / "toy_gallery.jsonl"
    manifest.write_text(
        "\n".join(
            [
                f'{{"image_id":"g/1","path":"{img1.as_posix()}","label":0}}',
                f'{{"image_id":"g/bad","path":"{bad.as_posix()}","label":1}}',
                f'{{"image_id":"g/2","path":"{img2.as_posix()}","label":2}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeEncoder:
        def __init__(self, **kwargs):
            from torchvision import transforms as T

            self.preprocess_val = T.Compose([T.Resize((224, 224)), T.ToTensor()])
            self.dino_preprocess = T.Compose([T.Resize((224, 224)), T.ToTensor()])
            self._bio_mean = [0.5, 0.5, 0.5]
            self._bio_std = [0.5, 0.5, 0.5]
            self._dino_mean = [0.5, 0.5, 0.5]
            self._dino_std = [0.5, 0.5, 0.5]

        def _encode_batch_tensors(self, bio_batch, dino_batch):
            B = bio_batch.shape[0]
            X = np.zeros((B, config.EMB_DIM), dtype=np.float32)
            for i in range(B):
                v1 = float(bio_batch[i, 0, 0, 0].item()) + 1.0
                v2 = float(bio_batch[i, 1, 0, -1].item()) + 1.0
                X[i, 0] = v1
                X[i, 1] = v2
                X[i, config.BIOCLIP_DIM] = v2
                X[i, config.BIOCLIP_DIM + 1] = v1
            return encoders._renorm_blocks(X)

        embed_manifest_shard = encoders.Encoder.embed_manifest_shard

    monkeypatch.setattr(encoders, "Encoder", FakeEncoder)
    out_npz = tmp_path / "toy_gallery.npz"
    encoders.main(["--manifest", str(manifest), "--out", str(out_npz), "--shard-size", "2"])

    es = contracts.EmbeddingSet.load(out_npz)
    assert es.ids == ["g/1", "g/2"]
    assert es.X.shape == (2, config.EMB_DIM)
    assert list(es.y) == [0, 2]
    contracts.check_embeddings(es.X)

    with np.load(out_npz, allow_pickle=False) as d:
        assert "X_aug" in d and "y_aug" in d
        assert d["X_aug"].shape == (4, config.EMB_DIM)
        assert list(d["y_aug"]) == [0, 0, 2, 2]
        contracts.check_embeddings(d["X_aug"])

