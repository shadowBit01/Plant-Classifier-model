"""Synthetic fixtures so every lane can test without data or a GPU."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vanalens.config import BIOCLIP_DIM, EMB_DIM
from vanalens.contracts import EmbeddingSet, Taxon

C = 5
TAXA = [
    Taxon(0, "Alpha one", "Alpha", "F1"),
    Taxon(1, "Alpha two", "Alpha", "F1"),
    Taxon(2, "Beta one", "Beta", "F1"),
    Taxon(3, "Gamma one", "Gamma", "F2"),
    Taxon(4, "Gamma two", "Gamma", "F2"),
]


def norm_blocks(X: np.ndarray) -> np.ndarray:
    a, b = X[:, :BIOCLIP_DIM], X[:, BIOCLIP_DIM:]
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.hstack([a, b]).astype(np.float32)


@pytest.fixture
def taxa() -> list[Taxon]:
    return [replace(t) for t in TAXA]   # fresh copies: tests may mutate gbif_key


@pytest.fixture
def clusters():
    """Gaussian class clusters in embedding space: gallery (20/class) and queries (10/class)."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(C, EMB_DIM))

    def draw(n_per: int, prefix: str):
        X = np.repeat(centers, n_per, 0) + 1.0 * rng.normal(size=(C * n_per, EMB_DIM))
        y = np.repeat(np.arange(C), n_per)
        ids = [f"{prefix}/{c}/{i}" for c in range(C) for i in range(n_per)]
        return norm_blocks(X), y, ids

    Xg, yg, idg = draw(20, "g")
    Xq, yq, _ = draw(10, "q")
    return {"gallery": EmbeddingSet(idg, Xg, yg), "Xq": Xq, "yq": yq, "centers": centers}


@pytest.fixture
def text_protos(clusters):
    """3 trait prototypes per class near the BioCLIP block of each class centre."""
    rng = np.random.default_rng(1)
    protos, strings = [], []
    for c in range(C):
        P = clusters["centers"][c, :BIOCLIP_DIM] + 0.3 * rng.normal(size=(3, BIOCLIP_DIM))
        protos.append((P / np.linalg.norm(P, axis=1, keepdims=True)).astype(np.float32))
        strings.append([f"trait {c}.{t}" for t in range(3)])
    return protos, strings


@pytest.fixture
def calibrated():
    """Probabilities that are calibrated by construction: y is sampled from each row."""
    rng = np.random.default_rng(2)
    P = rng.dirichlet(np.full(C, 0.5), size=6000)
    y = np.array([rng.choice(C, p=row) for row in P])
    return P, y
