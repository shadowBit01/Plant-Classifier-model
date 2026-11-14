"""Lane L3 (core-ml). Three scorers over frozen embeddings. All return (N, C) probabilities.

Tests: tests/test_core.py
"""
from __future__ import annotations

import faiss
import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import BIOCLIP_DIM
from .contracts import EmbeddingSet

EPS = 1e-6


def _l2norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


class KNNScorer:
    """Cosine kNN over the reference gallery (inner product on L2-normalised X; use faiss.IndexFlatIP).

    predict_proba: per query take top-k neighbours, add exp(sim / temperature) to each neighbour's class,
    add EPS to every class, renormalise. Shape (N, n_classes), column j = class idx j.
    neighbors: the top-n gallery image ids per query (evidence shown in the UI).
    add(): append new gallery items (few-shot "add a species") without rebuilding from scratch.
    """

    def __init__(self, gallery: EmbeddingSet, n_classes: int, k: int = 10, temperature: float = 0.05):
        if gallery.y is None:
            raise ValueError("KNNScorer needs a labelled gallery")
        self.n_classes = n_classes
        self.k = k
        self.temperature = temperature
        self.index = faiss.IndexFlatIP(gallery.X.shape[1])
        self.ids: list[str] = []
        self.y = np.empty(0, dtype=np.int64)
        self.add(gallery)

    def add(self, items: EmbeddingSet) -> None:
        if items.y is None:
            raise ValueError("added gallery items need labels")
        self.index.add(np.ascontiguousarray(items.X, dtype=np.float32))
        self.ids.extend(items.ids)
        self.y = np.concatenate([self.y, np.asarray(items.y, dtype=np.int64)])

    def _search(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        Xq = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float32)
        return self.index.search(Xq, min(k, self.index.ntotal))

    def max_similarity(self, X: np.ndarray) -> np.ndarray:
        """Best gallery cosine per query (input to conformal.is_unknown)."""
        D, _ = self._search(X, 1)
        return D[:, 0].astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        D, I = self._search(X, self.k)
        P = np.full((D.shape[0], self.n_classes), EPS, dtype=np.float64)
        valid = I >= 0
        rows = np.broadcast_to(np.arange(D.shape[0])[:, None], D.shape)[valid]
        np.add.at(P, (rows, self.y[I[valid]]), np.exp(D[valid].astype(np.float64) / self.temperature))
        return P / P.sum(1, keepdims=True)

    def neighbors(self, X: np.ndarray, n: int = 3) -> list[list[str]]:
        _, I = self._search(X, n)
        return [[self.ids[i] for i in row if i >= 0] for row in I]


class ProbeScorer:
    """Multinomial logistic regression on frozen features (sklearn, or a 1-layer torch head on GPU if slow).

    Output columns MUST be class idx 0..n_classes-1 even if some class has no training rows
    (give those columns EPS, then renormalise).
    """

    def __init__(self, n_classes: int, C: float = 1.0):
        self.n_classes = n_classes
        self.C = C
        self.model: LogisticRegression | None = None

    def fit(self, gallery: EmbeddingSet) -> "ProbeScorer":
        if gallery.y is None:
            raise ValueError("ProbeScorer needs a labelled gallery")
        self.model = LogisticRegression(C=self.C, max_iter=1000)
        self.model.fit(np.asarray(gallery.X, dtype=np.float32), np.asarray(gallery.y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        P = np.full((X.shape[0], self.n_classes), EPS, dtype=np.float64)
        P[:, self.model.classes_.astype(np.int64)] = self.model.predict_proba(X)
        return P / P.sum(1, keepdims=True)


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
        assert len(protos) == len(trait_strings)
        self.protos = [_l2norm(np.asarray(p, dtype=np.float64)) for p in protos]
        self.trait_strings = trait_strings
        self.temperature = temperature

    def _bio(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return _l2norm(X[:, :BIOCLIP_DIM])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        B = self._bio(X)
        S = np.stack([(B @ p.T).mean(1) for p in self.protos], axis=1) / self.temperature
        E = np.exp(S - S.max(1, keepdims=True))
        return E / E.sum(1, keepdims=True)

    def explain(self, x: np.ndarray, c: int, top: int = 3) -> list[str]:
        sims = (self._bio(x) @ self.protos[c].T)[0]
        return [self.trait_strings[c][i] for i in np.argsort(-sims)[:top]]
