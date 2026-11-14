"""Lane L3 (core-ml). Three scorers over frozen embeddings. All return (N, C) probabilities.

Tests: tests/test_core.py
"""
from __future__ import annotations

import numpy as np

from .config import BIOCLIP_DIM
from .contracts import EmbeddingSet

EPS = 1e-6


class KNNScorer:
    """Cosine kNN over the reference gallery (inner product on L2-normalised X; use faiss.IndexFlatIP).

    predict_proba: per query take top-k neighbours, add exp(sim / temperature) to each neighbour's class,
    add EPS to every class, renormalise. Shape (N, n_classes), column j = class idx j.
    neighbors: the top-n gallery image ids per query (evidence shown in the UI).
    add(): append new gallery items (few-shot "add a species") without rebuilding from scratch.
    """

    def __init__(self, gallery: EmbeddingSet, n_classes: int, k: int = 10, temperature: float = 0.05):
        raise NotImplementedError

    def add(self, items: EmbeddingSet) -> None:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def neighbors(self, X: np.ndarray, n: int = 3) -> list[list[str]]:
        raise NotImplementedError


class ProbeScorer:
    """Multinomial logistic regression on frozen features (sklearn, or a 1-layer torch head on GPU if slow).

    Output columns MUST be class idx 0..n_classes-1 even if some class has no training rows
    (give those columns EPS, then renormalise).
    """

    def __init__(self, n_classes: int, C: float = 1.0):
        raise NotImplementedError

    def fit(self, gallery: EmbeddingSet) -> "ProbeScorer":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TextScorer:
    """Zero-shot scorer over text prototypes (classification by description).

    protos[c]: (T_c, BIOCLIP_DIM) L2-normalised BioCLIP 2 text embeddings for class c
               (trait prompts + a plain "a photo of <species>" prompt).
    trait_strings[c]: the T_c strings behind protos[c], same order.
    Uses only the BioCLIP block X[:, :BIOCLIP_DIM] (re-normalise it).
    score[n, c] = mean_t cos(x_n, protos[c][t]);  P = softmax(score / temperature).
    explain(x, c, top): the `top` trait strings of class c with highest cosine to x.
    """

    def __init__(self, protos: list[np.ndarray], trait_strings: list[list[str]], temperature: float = 0.01):
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def explain(self, x: np.ndarray, c: int, top: int = 3) -> list[str]:
        raise NotImplementedError
