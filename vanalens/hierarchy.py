"""Lane L3 (core-ml). Taxonomic aggregation and backoff. Tests: tests/test_core.py"""
from __future__ import annotations

import numpy as np

from .contracts import Taxon


def aggregate(P: np.ndarray, taxa: list[Taxon], rank: str) -> tuple[np.ndarray, list[str]]:
    """Sum species columns into `rank` ("genus" or "family").
    Returns (P_rank of shape (N, R), names) with names sorted alphabetically; rows still sum to 1."""
    raise NotImplementedError


def backoff(p: np.ndarray, taxa: list[Taxon], tau: float = 0.6) -> tuple[str, str, float]:
    """p: (C,) one distribution. Return the deepest confident rank:
    ("species", species, prob) if max species prob >= tau, else ("genus", genus, prob) if max genus prob >= tau,
    else ("family", family, prob) if max family prob >= tau, else ("unknown", "", max family prob)."""
    raise NotImplementedError
