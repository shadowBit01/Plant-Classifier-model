"""Lane L3 (core-ml). Split conformal prediction sets. Tests: tests/test_core.py

Start with the standard score s = 1 - P[y]. Stretch goal: prevalence-adjusted score for the long tail
(Ding, Fermanian, Salmon 2025, arXiv 2507.06867), behind a flag.
"""
from __future__ import annotations

import math

import numpy as np


def fit_qhat(P_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1) -> float:
    """Scores s_i = 1 - P_cal[i, y_i]; return the ceil((n+1)(1-alpha))/n empirical quantile
    (np.quantile(..., method="higher")). Clip the level to <= 1."""
    y = np.asarray(y_cal, dtype=np.int64)
    n = len(y)
    s = 1.0 - P_cal[np.arange(n), y]
    level = min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(s, level, method="higher"))


def predict_sets(P: np.ndarray, qhat: float) -> list[np.ndarray]:
    """Per row: class idxs with P >= 1 - qhat, sorted by descending P. Never empty: include argmax."""
    thr = 1.0 - qhat
    out = []
    for p in np.atleast_2d(P):
        order = np.argsort(-p, kind="stable")
        keep = order[p[order] >= thr]
        out.append(keep if keep.size else order[:1])
    return out


def coverage(sets: list[np.ndarray], y: np.ndarray) -> float:
    """Fraction of rows whose true class is in its set."""
    return float(np.mean([int(t) in np.asarray(s).tolist() for s, t in zip(sets, y)]))


def mean_set_size(sets: list[np.ndarray]) -> float:
    return float(np.mean([len(s) for s in sets]))


def is_unknown(max_similarity: np.ndarray, threshold: float) -> np.ndarray:
    """Open-set reject: True where the best gallery cosine similarity is below `threshold`
    (threshold fitted on val so that ~95% of known-species queries pass)."""
    return np.asarray(max_similarity) < threshold
