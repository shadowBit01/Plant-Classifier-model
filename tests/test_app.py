"""Lane L5 tests: the UI builds and the mock pipeline honours the Prediction contract."""
from __future__ import annotations

import numpy as np
import pytest

from vanalens.contracts import RANKS, Taxon
from vanalens.pipeline import MockPipeline


def test_mock_prediction_contract():
    m = MockPipeline()
    for paths in (["a.jpg"], ["leaf.jpg", "flower.jpg"]):
        p = m.predict(paths)
        assert abs(p.probs.sum() - 1) < 1e-9 and p.pred_set and p.rank in RANKS
        assert p.top_k[0][0] == int(np.argmax(p.probs)) and len(p.evidence) == 3


def test_mock_add_species():
    m = MockPipeline()
    m.add_species(Taxon(-1, "Tectona grandis", "Tectona", "Lamiaceae"), ["x.jpg"])
    assert m.taxa[-1].idx == 5


def test_ui_builds(monkeypatch):
    pytest.importorskip("gradio")
    monkeypatch.setenv("VANALENS_MOCK", "1")
    from app.app import build_ui
    assert build_ui() is not None


def test_space_files_and_front_matter():
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    readme = (repo / "space" / "README.md").read_text(encoding="utf-8")
    reqs = (repo / "space" / "requirements.txt").read_text(encoding="utf-8")
    entry = repo / "space" / "app.py"
    assert entry.is_file()
    assert readme.startswith("---\n") and "\n---\n" in readme[4:]
    front_matter = readme.split("---", 2)[1]
    assert "sdk: gradio" in front_matter
    assert "app_file: app.py" in front_matter
    assert "gradio" in reqs and "faiss-cpu" in reqs


def test_space_loads_state_from_env_cache(tmp_path, monkeypatch, taxa, clusters):
    import json
    from vanalens import config as vanalens_config
    from vanalens.contracts import save_taxa
    from vanalens.pipeline import VanaLens

    cache_dir = tmp_path / "custom_cache"
    (cache_dir / "state").mkdir(parents=True)
    (cache_dir / "emb").mkdir(parents=True)
    (cache_dir / "inat_india").mkdir(parents=True)

    state = {
        "scorers": ["knn", "probe"],
        "weights": {"knn": 0.5, "probe": 0.5},
        "T": 0.85,
        "qhat": 0.42,
        "thr": 0.25,
        "beta": 0.1,
    }
    (cache_dir / "state" / "inat_india.json").write_text(json.dumps(state), encoding="utf-8")
    clusters["gallery"].save(cache_dir / "emb" / "inat_india_gallery.npz")
    save_taxa(taxa, cache_dir / "inat_india" / "labels.json")

    monkeypatch.delenv("VANALENS_MOCK", raising=False)
    monkeypatch.delenv("VANALENS_DATA", raising=False)
    monkeypatch.setenv("VANALENS_CACHE", str(cache_dir))
    monkeypatch.setattr(vanalens_config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(vanalens_config, "DATA_DIR", tmp_path / "nonexistent_data")

    import os
    import space.app as space_app

    space_app.configure_space_env()
    assert os.environ["VANALENS_MOCK"] == "0"
    pipeline = space_app.load_pipeline()
    assert isinstance(pipeline, VanaLens)
    assert abs(pipeline.T - 0.85) < 1e-6 and abs(pipeline.qhat - 0.42) < 1e-6
    assert len(pipeline.taxa) == len(taxa)

