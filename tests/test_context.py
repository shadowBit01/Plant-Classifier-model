"""Lane L4 tests: geo prior, traits, reranker (no network)."""
from __future__ import annotations

import hashlib

import numpy as np

from vanalens.geo import apply_prior, geo_prior
from vanalens.rerank import build_rerank_prompt, parse_choice, should_rerank
from vanalens.traits import build_text_protos, trait_prompts


def test_geo_prior(taxa):
    for i, t in enumerate(taxa):
        t.gbif_key = 100 + i
    taxa[4].gbif_key = None
    pr = geo_prior({100: 50, 101: 5}, taxa)
    assert pr.shape == (5,) and abs(pr.sum() - 1) < 1e-9
    assert pr[0] > pr[1] > pr[2] and abs(pr[2] - pr[4]) < 1e-12
    assert np.allclose(geo_prior({}, taxa), 0.2)


def test_apply_prior():
    P = np.array([[0.5, 0.5, 0.0 + 1e-9]])
    P = P / P.sum(1, keepdims=True)
    prior = np.array([0.8, 0.1, 0.1])
    assert np.allclose(apply_prior(P, prior, beta=0.0), P)
    out = apply_prior(P, prior, beta=0.5)
    assert abs(out.sum() - 1) < 1e-9 and out[0, 0] > out[0, 1]


def _fake_encode(strings):
    v = np.stack([np.frombuffer(hashlib.sha256(s.encode()).digest() * 24, dtype=np.uint8)[:768].astype(float)
                  for s in strings]) - 127.5
    return v.astype(np.float32)


def test_trait_prompts():
    p = trait_prompts("Ficus religiosa", ["heart-shaped leaf", "long drip tip"])
    assert p[-1] == "a photo of Ficus religiosa" and len(p) == 3
    assert p[0] == "a photo of Ficus religiosa, which has heart-shaped leaf"


def test_build_text_protos(taxa):
    traits = {"Alpha one": ["a", "b"], "Beta one": ["c"]}
    protos, strings = build_text_protos(taxa, traits, _fake_encode)
    assert len(protos) == len(taxa) == len(strings)
    assert protos[0].shape == (3, 768) and protos[1].shape == (1, 768)
    assert all(np.allclose(np.linalg.norm(p, axis=1), 1, atol=1e-5) for p in protos)


def test_should_rerank():
    assert not should_rerank([3]) and should_rerank([3, 1])


def test_prompt_and_parse(taxa):
    cands = taxa[:3]
    prompt = build_rerank_prompt(cands, {"Alpha one": ["x"]})
    assert all(t.species in prompt for t in cands) and "ANSWER:" in prompt
    assert parse_choice("reasoning...\nANSWER:  alpha   TWO.", cands) == 1
    assert parse_choice("ANSWER: Beta one\nchanged mind\nANSWER: Alpha one", cands) == 0
    assert parse_choice("I think it is Gamma one", cands) is None
    assert parse_choice("ANSWER: Gamma one", cands) is None


def test_gbif_species_counts_cache(tmp_path):
    from vanalens.geo import gbif_species_counts
    cache_file = tmp_path / "23.0_80.0_50_6.json"
    cache_file.write_text('{"123": 45, "678": 90}')
    counts = gbif_species_counts(23.18, 79.99, radius_km=50, kingdom_key=6, cache_dir=tmp_path)
    assert counts == {123: 45, 678: 90}


def test_resolve_gbif_keys_cache(tmp_path, taxa):
    from vanalens.geo import resolve_gbif_keys
    cache_file = tmp_path / "species_keys.json"
    cache_file.write_text('{"Alpha one": 99999}')
    resolve_gbif_keys(taxa[:1], cache_dir=tmp_path)
    assert taxa[0].gbif_key == 99999


def test_generate_traits_fallback(tmp_path, taxa):
    from vanalens.traits import generate_traits
    cache_path = tmp_path / "traits.json"
    traits = generate_traits(taxa[:2], cache_path=cache_path)
    assert len(traits) == 2
    assert all(len(v) == 6 for v in traits.values())
    assert cache_path.exists()


def test_rerank_disabled_no_key(taxa):
    from vanalens.rerank import rerank
    cands = taxa[:3]
    chosen, rationale, reranked = rerank([], cands, {}, {})
    assert chosen == cands[0].idx
    assert not reranked
    assert "no key" in rationale


def test_rerank_with_mock_client(taxa):
    from unittest.mock import MagicMock
    from vanalens.rerank import rerank

    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text="Looking at photo...\nANSWER: Alpha two")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg

    cands = taxa[:3]
    chosen, rationale, reranked = rerank([], cands, {}, {}, client=mock_client)
    assert chosen == 1
    assert reranked

