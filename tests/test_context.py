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
