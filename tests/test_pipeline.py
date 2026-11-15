"""Lane L6: VanaLens integration on synthetic fixtures (encoder monkeypatched; no GPU, network or cache)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import C, norm_blocks
from vanalens import config, geo, pipeline
from vanalens.config import BIOCLIP_DIM, EMB_DIM
from vanalens.contracts import EmbeddingSet, Prediction, Taxon, save_taxa
from vanalens.pipeline import MockPipeline, VanaLens

STATE = {
    "dataset": "toy", "n_classes": C, "scorers": ["knn", "probe", "text"],
    "weights": {"knn": 0.3, "probe": 0.5, "text": 0.2}, "T": 1.0, "alpha": 0.1, "qhat": 0.5, "thr": 0.0,
    "knn": {"k": 10, "temperature": 0.05}, "probe": {"C": 10.0}, "text": {"temperature": 0.01},
}


class FakeEncoder:
    """Maps a path to a precomputed embedding row; encode_text gives random unit vectors."""

    def __init__(self, table: dict[str, np.ndarray]):
        self.table = table
        self.rng = np.random.default_rng(5)

    def embed_paths(self, paths):
        return np.vstack([self.table[p] for p in paths]).astype(np.float32)

    def encode_text(self, strings):
        V = self.rng.normal(size=(len(strings), BIOCLIP_DIM))
        return (V / np.linalg.norm(V, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def lens(taxa, clusters, text_protos, monkeypatch):
    table = {f"q{i}.jpg": clusters["Xq"][i] for i in range(len(clusters["Xq"]))}
    monkeypatch.setattr(pipeline, "_load_encoder", lambda: FakeEncoder(table))
    monkeypatch.delenv(config.LLM_KEY_ENV, raising=False)
    return VanaLens(taxa, clusters["gallery"], dict(STATE), text=text_protos)


def test_predict_shape_and_accuracy(lens, clusters):
    yq = clusters["yq"]
    correct = 0
    for i in range(0, len(yq), 3):
        pr = lens.predict([f"q{i}.jpg"])
        assert isinstance(pr, Prediction)
        assert pr.probs.shape == (C,) and np.isclose(pr.probs.sum(), 1.0)
        assert pr.pred_set and pr.top_k[0][0] == int(np.argmax(pr.probs))
        assert pr.rank in ("species", "genus", "family", "unknown")
        assert len(pr.evidence) == 3 and all(e.startswith("g/") for e in pr.evidence)
        assert len(pr.traits) == 3 and pr.organ in pipeline.ORGAN_PROMPTS
        assert not pr.reranked                       # no key -> reranker skipped
        correct += pr.top_k[0][0] == yq[i]
    assert correct / len(range(0, len(yq), 3)) >= 0.9


def test_multi_photo_sharpens(lens, clusters):
    yq = clusters["yq"]
    idx = np.where(yq == 2)[0][:3]
    one = lens.predict([f"q{idx[0]}.jpg"])
    many = lens.predict([f"q{i}.jpg" for i in idx])
    assert many.top_k[0][0] == 2
    assert many.probs[2] >= one.probs[2] - 1e-9


def test_text_scorer_skipped_without_protos(taxa, clusters):
    v = VanaLens(taxa, clusters["gallery"], dict(STATE), text=None, encoder=FakeEncoder({}))
    assert v.names == ["knn", "probe"] and v.text is None
    P = v.scores(clusters["Xq"])
    assert P.shape == (len(clusters["Xq"]), C) and np.allclose(P.sum(1), 1)


def test_open_set_gate(taxa, clusters, monkeypatch):
    rng = np.random.default_rng(9)
    far = norm_blocks(rng.normal(size=(1, EMB_DIM)))[0]
    state = dict(STATE, thr=1.5)                     # stricter than any random query can reach
    v = VanaLens(taxa, clusters["gallery"], state, encoder=FakeEncoder({"x.jpg": far}))
    pr = v.predict(["x.jpg"])
    assert pr.rank == "unknown" and "open-set" in pr.rationale


def test_geo_prior_applied_once(lens, clusters, monkeypatch):
    lens.beta = 1.0
    prior = np.array([0.02, 0.02, 0.02, 0.02, 0.92])
    monkeypatch.setattr(geo, "gbif_species_counts", lambda lat, lon, **k: {1: 1})
    monkeypatch.setattr(geo, "geo_prior", lambda counts, taxa, **k: prior)
    i = int(np.where(clusters["yq"] == 0)[0][0])
    base = lens.predict([f"q{i}.jpg"])
    with_geo = lens.predict([f"q{i}.jpg", f"q{i}.jpg"], lat=20.0, lon=78.0)
    two = lens.predict([f"q{i}.jpg", f"q{i}.jpg"])
    expect = two.probs * prior
    assert np.allclose(with_geo.probs, expect / expect.sum(), atol=1e-6)
    assert with_geo.probs[4] > two.probs[4] and np.isclose(base.probs.sum(), 1)


def test_rerank_called_when_key_and_set(lens, clusters, monkeypatch):
    monkeypatch.setenv(config.LLM_KEY_ENV, "test-key")
    lens.qhat = 1.0                                   # set = all classes -> reranker triggers
    seen = {}

    def fake_rerank(paths, cands, refs, traits, client=None):
        seen["n"] = len(cands)
        return cands[-1].idx, "ANSWER: last", True

    monkeypatch.setattr(pipeline.rerank, "rerank", fake_rerank)
    pr = lens.predict(["q0.jpg"])
    assert pr.reranked and seen["n"] == 5 and pr.rank == "species"
    assert pr.label == lens.taxa[pr.pred_set[4]].species


def test_add_species_few_shot(lens, clusters):
    rng = np.random.default_rng(3)
    center = rng.normal(size=EMB_DIM)
    new = norm_blocks(center + 1.0 * rng.normal(size=(8, EMB_DIM)))
    for k in range(8):
        lens.encoder.table[f"new{k}.jpg"] = new[k]
    lens.add_species(Taxon(99, "Delta one", "Delta", "F3"), [f"new{k}.jpg" for k in range(5)])
    assert len(lens.taxa) == C + 1 and lens.taxa[-1].idx == C
    assert len(lens.text.protos) == C + 1
    pr = lens.predict(["new6.jpg"])
    assert pr.probs.shape == (C + 1,) and pr.top_k[0][0] == C
    assert pr.evidence[0].startswith("added/Delta one/")
    old = lens.predict(["q0.jpg"])                    # old classes still work
    assert old.top_k[0][0] == clusters["yq"][0]


def test_fit_beta_prefers_informative_prior(lens, clusters, monkeypatch):
    yq = clusters["yq"]
    # prior concentrated on the true class -> beta > 0 lowers NLL
    monkeypatch.setattr(geo, "gbif_species_counts", lambda lat, lon, **k: {1: 1})
    monkeypatch.setattr(geo, "geo_prior", lambda counts, taxa, **k: np.full(C, 1.0 / C))
    b0, t0 = pipeline.fit_beta(lens, clusters["Xq"], yq, [(1.0, 1.0)] * len(yq))
    assert b0 == 0.0 and len(set(t0.values())) == 1   # uniform prior: all equal, ties -> smallest beta
    priors = iter([np.where(np.arange(C) == y, 0.8, 0.05) for y in yq])
    monkeypatch.setattr(geo, "geo_prior", lambda counts, taxa, **k: next(priors))
    b1, t1 = pipeline.fit_beta(lens, clusters["Xq"], yq, [(1.0, 1.0)] * len(yq))
    assert b1 > 0 and t1[str(b1)] < t1["0.0"]


def _write_cache(tmp_path, taxa, clusters, text_protos, with_text: bool):
    data, cache = tmp_path / "data", tmp_path / "cache"
    for d in (data / "toy", cache / "emb", cache / "state", cache / "text", cache / "manifests"):
        d.mkdir(parents=True, exist_ok=True)
    save_taxa(taxa, data / "toy" / "labels.json")
    clusters["gallery"].save(cache / "emb" / "toy_gallery.npz")
    yq = clusters["yq"]
    ids = [f"cal/{i}" for i in range(len(yq))]
    EmbeddingSet(ids, clusters["Xq"], yq).save(cache / "emb" / "toy_cal.npz")
    lines = [json.dumps({"image_id": i, "path": f"{i}.jpg", "label": int(y), "lat": 10.0, "lon": 20.0})
             for i, y in zip(ids, yq)]
    (cache / "manifests" / "toy_cal.jsonl").write_text("\n".join(lines) + "\n")
    (cache / "state" / "toy.json").write_text(json.dumps(STATE))
    if with_text:
        protos, strings = text_protos
        np.savez(cache / "text" / "toy.npz", stacked_protos=np.stack(protos), counts=np.array([3] * C),
                 species=np.array([t.species for t in taxa]), strings_json=json.dumps(strings))
    return data, cache


@pytest.mark.parametrize("with_text", [True, False])
def test_from_cache_and_fit_beta_cached(tmp_path, taxa, clusters, text_protos, monkeypatch, with_text):
    data, cache = _write_cache(tmp_path, taxa, clusters, text_protos, with_text)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "CACHE_DIR", cache)
    monkeypatch.setattr(geo, "gbif_species_counts", lambda lat, lon, **k: {})   # no network
    v = VanaLens.from_cache("toy")
    assert (v.text is not None) == with_text
    assert ("text" in v.names) == with_text and v.beta == 0.0
    out = pipeline.fit_beta_cached("toy")
    st = json.loads((cache / "state" / "toy.json").read_text())
    assert st["beta"] == out["beta"] == 0.0 and st["beta_fit"]["n_with_coords"] == len(clusters["yq"])
    assert st["qhat"] == STATE["qhat"]               # L3 fields untouched
    assert VanaLens.from_cache("toy").beta == st["beta"]


def test_from_cache_missing_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        VanaLens.from_cache("nope")


def test_mock_pipeline_unchanged():
    m = MockPipeline()
    pr = m.predict(["a.jpg"])
    assert pr.probs.shape == (5,) and pr.pred_set
    m.add_species(Taxon(0, "X y", "X", "Z"), ["b.jpg"])
    assert len(m.taxa) == 6


def test_get_pipeline_mock(monkeypatch):
    monkeypatch.setenv("VANALENS_MOCK", "1")
    assert isinstance(pipeline.get_pipeline(), MockPipeline)
