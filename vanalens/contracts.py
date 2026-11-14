"""Shared data contracts. FROZEN: every lane codes against this file. Ask the human before changing.

Conventions
-----------
- Class index j in every (N, C) matrix == Taxon.idx == position in labels.json.
- Probability matrices: float, shape (N, C), each row sums to 1 (check_proba).
- Embeddings: float32, shape (N, EMB_DIM) = [bioclip (BIOCLIP_DIM) | dino (DINO_DIM)],
  each block L2-normalised on its own (check_embeddings).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import BIOCLIP_DIM, EMB_DIM

ORGANS = ("leaf", "flower", "fruit", "bark", "whole_plant", "other")
RANKS = ("species", "genus", "family", "unknown")


@dataclass
class Taxon:
    idx: int
    species: str            # "Ficus religiosa"
    genus: str              # "Ficus"
    family: str             # "Moraceae"
    gbif_key: int | None = None


@dataclass
class Sample:
    image_id: str           # unique across all datasets, e.g. "dimpsar/leaf/0001"
    path: str               # absolute path to JPEG/PNG
    label: int | None       # Taxon.idx, or None if unlabelled
    source: str = ""        # "dimpsar" | "plantnet300k" | "campus" | "inat"
    group: str = ""         # split unit: observation / plant / video id. Never split a group.
    organ: str | None = None
    lat: float | None = None
    lon: float | None = None


@dataclass
class EmbeddingSet:
    ids: list[str]
    X: np.ndarray           # (N, EMB_DIM) float32
    y: np.ndarray | None    # (N,) int64 or None
    encoder: str = "bioclip2+dinov3"

    def save(self, path: str | Path) -> None:
        y = self.y if self.y is not None else np.array([], dtype=np.int64)
        np.savez(path, ids=np.array(self.ids), X=self.X.astype(np.float32), y=y, encoder=self.encoder)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingSet":
        d = np.load(path, allow_pickle=False)
        y = d["y"] if d["y"].size else None
        return cls(ids=[str(i) for i in d["ids"]], X=d["X"], y=y, encoder=str(d["encoder"]))


@dataclass
class Prediction:
    probs: np.ndarray                    # (C,) final distribution after fusion + priors
    top_k: list[tuple[int, float]]       # [(class idx, prob)], descending
    pred_set: list[int]                  # conformal set (class idxs), never empty
    rank: str                            # one of RANKS
    label: str                           # display name at `rank`
    confidence: float                    # probability mass behind `label`
    evidence: list[str] = field(default_factory=list)   # nearest gallery image_ids
    traits: list[str] = field(default_factory=list)     # matched trait strings
    organ: str | None = None
    reranked: bool = False
    rationale: str = ""


def save_taxa(taxa: list[Taxon], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(t) for t in taxa], indent=1))


def load_taxa(path: str | Path) -> list[Taxon]:
    taxa = [Taxon(**d) for d in json.loads(Path(path).read_text())]
    assert [t.idx for t in taxa] == list(range(len(taxa))), "labels.json idx must be 0..C-1 in order"
    return taxa


def check_proba(P: np.ndarray, n_classes: int, atol: float = 1e-4) -> None:
    assert P.ndim == 2 and P.shape[1] == n_classes, f"expected (N, {n_classes}), got {P.shape}"
    assert np.all(P >= -1e-9), "negative probabilities"
    assert np.allclose(P.sum(1), 1.0, atol=atol), "rows must sum to 1"


def check_embeddings(X: np.ndarray, atol: float = 1e-3) -> None:
    assert X.ndim == 2 and X.shape[1] == EMB_DIM, f"expected (N, {EMB_DIM}), got {X.shape}"
    assert X.dtype == np.float32, f"expected float32, got {X.dtype}"
    for blk in (X[:, :BIOCLIP_DIM], X[:, BIOCLIP_DIM:]):
        assert np.allclose(np.linalg.norm(blk, axis=1), 1.0, atol=atol), "each block must be L2-normalised"
