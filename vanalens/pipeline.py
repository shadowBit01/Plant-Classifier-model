"""Lane L6 (integration) owns VanaLens. MockPipeline is implemented now so the app lane can start at hour 0.

Predict flow (VanaLens.predict):
  1 organ = zero-shot BioCLIP 2 over ORGANS; reject "other" as not-a-plant
  2 X = encoder.embed_paths(paths)                 (one row per photo)
  3 P = fusion.fuse([knn, probe, text], weights)   then fusion.apply_temperature(P, T)
  4 if lat/lon: P = geo.apply_prior(P, geo.geo_prior(gbif_species_counts(...), taxa), beta)
  5 multi-photo: p = softmax(sum over photos of log P)
  6 set = conformal.predict_sets(p, qhat); unknown if conformal.is_unknown(max sim, thr)
  7 rank, label, conf = hierarchy.backoff(p, taxa, BACKOFF_TAU)
  8 if rerank.should_rerank(set): rerank over top-5 of set
  9 evidence = knn.neighbors(...); traits = text.explain(...)
Fitted state (weights, T, qhat, thr, beta) is saved to CACHE_DIR/state/<dataset>.json by the integration lane.
"""
from __future__ import annotations

import os

import numpy as np

from .contracts import Prediction, Taxon


class VanaLens:
    @classmethod
    def from_cache(cls, dataset: str) -> "VanaLens":
        raise NotImplementedError

    def predict(self, paths: list[str], lat: float | None = None, lon: float | None = None,
                use_rerank: bool = True) -> Prediction:
        raise NotImplementedError

    def add_species(self, taxon: Taxon, paths: list[str]) -> None:
        """Few-shot: embed `paths`, append to the kNN gallery, add a text prototype. No retraining."""
        raise NotImplementedError


class MockPipeline:
    """Deterministic fake predictions with the real Prediction shape, for UI work before the model exists."""

    TAXA = [
        Taxon(0, "Ficus religiosa", "Ficus", "Moraceae"),
        Taxon(1, "Ficus benghalensis", "Ficus", "Moraceae"),
        Taxon(2, "Azadirachta indica", "Azadirachta", "Meliaceae"),
        Taxon(3, "Ocimum tenuiflorum", "Ocimum", "Lamiaceae"),
        Taxon(4, "Mangifera indica", "Mangifera", "Anacardiaceae"),
    ]

    def __init__(self):
        self.taxa = list(self.TAXA)

    def predict(self, paths: list[str], lat: float | None = None, lon: float | None = None,
                use_rerank: bool = True) -> Prediction:
        seed = sum(len(p) for p in paths) + len(paths)
        rng = np.random.default_rng(seed)
        logits = rng.normal(size=len(self.taxa)) * (1.5 + len(paths))
        p = np.exp(logits - logits.max()); p /= p.sum()
        order = np.argsort(-p)
        pred_set = [int(i) for i in order if p[i] >= 0.1] or [int(order[0])]
        top = self.taxa[int(order[0])]
        if p[order[0]] >= 0.6:
            rank, label, conf = "species", top.species, float(p[order[0]])
        else:
            rank, label = "genus", top.genus
            conf = float(sum(p[t.idx] for t in self.taxa if t.genus == top.genus))
        return Prediction(
            probs=p, top_k=[(int(i), float(p[i])) for i in order[:5]], pred_set=pred_set,
            rank=rank, label=label, confidence=conf,
            evidence=[f"mock/{top.species.replace(' ', '_')}/{k}.jpg" for k in range(3)],
            traits=["ovate leaf with long drip tip", "entire margin", "prominent pale midrib"],
            organ="leaf", reranked=len(pred_set) > 1 and use_rerank,
            rationale="mock: drip tip and midrib match the top candidate" if len(pred_set) > 1 else "",
        )

    def add_species(self, taxon: Taxon, paths: list[str]) -> None:
        self.taxa.append(Taxon(len(self.taxa), taxon.species, taxon.genus, taxon.family))


def get_pipeline(dataset: str = "campus"):
    """VANALENS_MOCK=1 -> MockPipeline, else the real VanaLens."""
    if os.environ.get("VANALENS_MOCK") == "1":
        return MockPipeline()
    return VanaLens.from_cache(dataset)
