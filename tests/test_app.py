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
