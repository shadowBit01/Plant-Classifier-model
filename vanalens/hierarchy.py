"""Lane L3 (core-ml). Taxonomic aggregation and backoff. Tests: tests/test_core.py"""
from __future__ import annotations

import numpy as np

from .contracts import Taxon


def aggregate(P: np.ndarray, taxa: list[Taxon], rank: str) -> tuple[np.ndarray, list[str]]:
    """Sum species columns into `rank` ("genus" or "family").
    Returns (P_rank of shape (N, R), names) with names sorted alphabetically; rows still sum to 1."""
    if rank not in ("genus", "family"):
        raise ValueError(f"rank must be 'genus' or 'family', got {rank!r}")
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    groups = [getattr(t, rank) for t in taxa]
    names = sorted(set(groups))
    pos = {n: i for i, n in enumerate(names)}
    M = np.zeros((len(taxa), len(names)))
    M[np.arange(len(taxa)), [pos[g] for g in groups]] = 1.0
    return P @ M, names


def backoff(p: np.ndarray, taxa: list[Taxon], tau: float = 0.6) -> tuple[str, str, float]:
    """p: (C,) one distribution. Return the deepest confident rank:
    ("species", species, prob) if max species prob >= tau, else ("genus", genus, prob) if max genus prob >= tau,
    else ("family", family, prob) if max family prob >= tau, else ("unknown", "", max family prob)."""
    p = np.asarray(p, dtype=np.float64)
    j = int(p.argmax())
    if p[j] >= tau:
        return "species", taxa[j].species, float(p[j])
    best = 0.0
    for rank in ("genus", "family"):
        Pr, names = aggregate(p, taxa, rank)
        r = int(Pr[0].argmax())
        best = float(Pr[0, r])
        if best >= tau:
            return rank, names[r], best
    return "unknown", "", best
