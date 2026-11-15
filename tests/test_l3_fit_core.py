"""Lane L3: scripts/fit_core.py end-to-end on synthetic embeddings (no data, no GPU)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from vanalens.config import BIOCLIP_DIM, EMB_DIM
from vanalens.contracts import EmbeddingSet, Taxon, save_taxa

ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location("fit_core", ROOT / "scripts" / "fit_core.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm(X):
    X = X.astype(np.float32)
    for sl in (slice(0, BIOCLIP_DIM), slice(BIOCLIP_DIM, EMB_DIM)):
        X[:, sl] /= np.linalg.norm(X[:, sl], axis=1, keepdims=True)
    return X


def test_fit_core_writes_state_and_ablation(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    C = 4
    centers = rng.normal(size=(C, EMB_DIM))
    data, cache, runs = tmp_path / "data", tmp_path / "cache", tmp_path / "runs"
    (data / "toy").mkdir(parents=True)
    (cache / "emb").mkdir(parents=True)
    (cache / "text").mkdir(parents=True)
    save_taxa([Taxon(i, f"G{i // 2} s{i}", f"G{i // 2}", "F") for i in range(C)], data / "toy" / "labels.json")
    for split, n in (("gallery", 15), ("cal", 15), ("test", 10)):
        y = np.repeat(np.arange(C), n)
        X = _norm(centers[y] + 0.6 * rng.normal(size=(len(y), EMB_DIM)))
        EmbeddingSet([f"toy/{split}/{i}" for i in range(len(y))], X, y).save(cache / "emb" / f"toy_{split}.npz")
    # Text protos for only 3 of 4 species: the 4th must get the neutral fallback.
    protos = np.stack([np.stack([centers[c, :BIOCLIP_DIM]] * 2) for c in range(3)]).astype(np.float32)
    protos /= np.linalg.norm(protos, axis=-1, keepdims=True)
    strings = [[f"t{c}a", f"t{c}b"] for c in range(3)]
    np.savez(cache / "text" / "toy.npz", stacked_protos=protos, counts=np.array([2, 2, 2], dtype=np.int32),
             species=np.array([f"G{i // 2} s{i}" for i in range(3)]), strings_json=json.dumps(strings))

    mod = _load_script()
    monkeypatch.setattr(mod, "DATA_DIR", data)
    monkeypatch.setattr(mod, "CACHE_DIR", cache)
    monkeypatch.setattr(mod, "RUNS_DIR", runs)
    assert mod.main(["--dataset", "toy"]) == 0

    state = json.loads((cache / "state" / "toy.json").read_text())
    assert state["scorers"] == ["knn", "probe", "text"]
    assert abs(sum(state["weights"].values()) - 1.0) < 1e-6
    assert state["T"] > 0 and 0 <= state["qhat"] <= 1
    assert state["text"]["missing_species"] == ["G1 s3"]
    abl = json.loads((runs / "L3" / "ablation_toy.json").read_text())
    assert list(abl["rows"]) == ["knn", "probe", "text", "fused", "fused+T", "fused+T+sets"]
    sets_row = abl["rows"]["fused+T+sets"]
    assert sets_row["top1"] > 0.8 and sets_row["coverage"] >= sets_row["top1"] and sets_row["mean_set_size"] >= 1
