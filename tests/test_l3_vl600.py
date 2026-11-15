"""Lane L3 (L3-c): torch heads, X_aug training, MLP / ensemble selection in fit_core.py. Synthetic, CPU only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from vanalens.config import BIOCLIP_DIM, EMB_DIM
from vanalens.contracts import EmbeddingSet, Taxon, check_proba, save_taxa
from vanalens.scorers import HeadScorer

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent


def _norm(X):
    X = X.astype(np.float32)
    for sl in (slice(0, BIOCLIP_DIM), slice(BIOCLIP_DIM, EMB_DIM)):
        X[:, sl] /= np.linalg.norm(X[:, sl], axis=1, keepdims=True)
    return X


def _blobs(rng, centers, n, noise=0.6):
    y = np.repeat(np.arange(len(centers)), n)
    return _norm(centers[y] + noise * rng.normal(size=(len(y), centers.shape[1]))), y


@pytest.mark.parametrize("hidden", [0, 64])
def test_head_learns_and_roundtrips(tmp_path, hidden):
    rng = np.random.default_rng(1)
    C = 5
    centers = rng.normal(size=(C, EMB_DIM))
    X, y = _blobs(rng, centers, 20)
    Xv, yv = _blobs(rng, centers, 10)
    # Class 5 has no training rows: its column must still exist (n_classes=6).
    h = HeadScorer(C + 1, hidden=hidden, dropout=0.2, epochs=15, patience=5, device="cpu").fit(X, y, Xv, yv)
    P = h.predict_proba(Xv)
    check_proba(P, C + 1)
    assert (P.argmax(1) == yv).mean() > 0.9
    assert 0 <= h.best_epoch < 15
    h.save(tmp_path / "head.npz")
    h2 = HeadScorer.load(tmp_path / "head.npz")
    assert h2.hidden == hidden and h2.best_epoch == h.best_epoch
    np.testing.assert_allclose(h2.predict_proba(Xv), P, atol=1e-6)


def test_head_options_without_val():
    rng = np.random.default_rng(2)
    centers = rng.normal(size=(3, EMB_DIM))
    X, y = _blobs(rng, centers, 15)
    h = HeadScorer(3, weight_decay=1e-4, label_smoothing=0.1, balanced=True, epochs=5, device="cpu").fit(X, y)
    assert h.best_epoch == 4
    assert (h.predict_proba(X).argmax(1) == y).mean() > 0.9


def _load_script():
    spec = importlib.util.spec_from_file_location("fit_core_l3c", ROOT / "scripts" / "fit_core.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_grid_expands_all_combinations():
    mod = _load_script()
    g = mod._grid({"a": (1, 2), "b": (True, False), "c": (0.0,)})
    assert len(g) == 4 and {"a": 2, "b": False, "c": 0.0} in g


def test_fit_core_vl600_like(tmp_path, monkeypatch):
    """Gallery with X_aug, no text protos: heads train on X + X_aug, mlp and ensemble rows, choices recorded."""
    rng = np.random.default_rng(3)
    C = 6
    centers = rng.normal(size=(C, EMB_DIM))
    data, cache, runs = tmp_path / "data", tmp_path / "cache", tmp_path / "runs"
    (data / "vlx").mkdir(parents=True)
    (cache / "emb").mkdir(parents=True)
    save_taxa([Taxon(i, f"G{i // 2} s{i}", f"G{i // 2}", f"F{i // 4}") for i in range(C)], data / "vlx" / "labels.json")
    for split, n in (("gallery", 12), ("cal", 10), ("test", 8)):
        X, y = _blobs(rng, centers, n, noise=0.7)
        ids = [f"vlx/{split}/{i}" for i in range(len(y))]
        path = cache / "emb" / f"vlx_{split}.npz"
        if split == "gallery":
            Xa, ya = _blobs(rng, centers, 2 * n, noise=0.7)
            np.savez(path, ids=np.array(ids), X=X, y=y, encoder="t", X_plain=X, X_aug=Xa, y_aug=ya)
        else:
            EmbeddingSet(ids, X, y).save(path)

    mod = _load_script()
    monkeypatch.setattr(mod, "DATA_DIR", data)
    monkeypatch.setattr(mod, "CACHE_DIR", cache)
    monkeypatch.setattr(mod, "RUNS_DIR", runs)
    monkeypatch.setattr(mod, "LINEAR_GRID", {"weight_decay": (0.0, 1e-4), "label_smoothing": (0.0,),
                                             "balanced": (False,)})
    monkeypatch.setattr(mod, "MLP_GRID", {"weight_decay": (0.0,), "label_smoothing": (0.0, 0.1),
                                          "balanced": (True,)})
    monkeypatch.setattr(mod, "LINEAR_FIXED", mod.LINEAR_FIXED | {"epochs": 10})
    monkeypatch.setattr(mod, "MLP_FIXED", mod.MLP_FIXED | {"hidden": 32, "epochs": 10})
    assert mod.main(["--dataset", "vlx", "--device", "cpu"]) == 0

    state = json.loads((cache / "state" / "vlx.json").read_text())
    assert state["train_rows"] == {"gallery": 72, "aug": 144}
    assert state["scorers"] == ["knn", "probe"]          # no text protos
    ct = state["default_head"]["cal_top1"]
    expect = "mlp" if ct["mlp"] > ct["probe"] + mod.MLP_MARGIN else "linear"
    assert state["probe"]["head"] == state["default_head"]["choice"] == expect
    for n in ("probe", "mlp"):
        assert Path(state["heads"][n]["weights"]).exists()
    ens = state["ensemble"]
    assert ens["scorers"] == ["knn", "probe", "mlp"] and abs(sum(ens["weights"].values()) - 1) < 1e-6
    assert ens["kept"] == (ens["cal_top1"] > ens["fused_cal_top1"])
    assert state["final"] == ("ensemble" if ens["kept"] else "fused")

    abl = json.loads((runs / "L3" / "ablation_vlx.json").read_text())
    assert list(abl["rows"]) == ["knn", "probe", "fused", "fused+T", "fused+T+sets", "mlp", "ensemble"]
    assert len(abl["head_grids"]["probe"]) == 2 and len(abl["head_grids"]["mlp"]) == 2
    for m in abl["rows"].values():
        for k in ("top1", "top5", "macro_top1", "genus_top1", "family_top1", "nll", "ece", "coverage",
                  "mean_set_size", "cal_top1"):
            assert k in m
    assert abl["rows"]["ensemble"]["top1"] > 0.8
    assert abl["final_row"] in abl["rows"]
