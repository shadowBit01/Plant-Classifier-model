"""Lane L3 (core-ml). Combine scorer outputs and calibrate. Tests: tests/test_core.py"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np


def _softmax(Z: np.ndarray) -> np.ndarray:
    E = np.exp(Z - Z.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


def fuse(probs: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Log-linear pooling: softmax(sum_i w_i * log(P_i + 1e-9)) row-wise. Weights need not sum to 1.
    A weight of 0 removes that scorer. Returns (N, C), rows sum to 1."""
    assert len(probs) == len(weights) and len(probs) > 0
    Z = np.zeros(np.shape(probs[0]), dtype=np.float64)
    for P, w in zip(probs, weights):
        if w != 0:
            Z = Z + w * np.log(np.asarray(P, dtype=np.float64) + 1e-9)
    return _softmax(Z)


def nll(P: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class (clip P at 1e-12)."""
    y = np.asarray(y, dtype=np.int64)
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean())


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    """softmax(log(P + 1e-12) / T) row-wise. T > 1 softens, T < 1 sharpens."""
    return _softmax(np.log(np.asarray(P, dtype=np.float64) + 1e-12) / T)


def fit_temperature(P: np.ndarray, y: np.ndarray, grid: Sequence[float] | None = None) -> float:
    """Return the T (default grid: 48 log-spaced values in [0.2, 5]) minimising nll(apply_temperature(P, T), y)."""
    if grid is None:
        grid = np.logspace(np.log10(0.2), np.log10(5.0), 48)
    losses = [nll(apply_temperature(P, T), y) for T in grid]
    return float(grid[int(np.argmin(losses))])


def _simplex_grid(m: int, n: int) -> Iterator[tuple[int, ...]]:
    """All m-tuples of non-negative ints summing to n."""
    if m == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in _simplex_grid(m - 1, n - i):
            yield (i,) + rest


def fit_weights(probs: Sequence[np.ndarray], y: np.ndarray, step: float = 0.1) -> tuple[float, ...]:
    """Grid-search weights on the simplex (step `step`) maximising top-1 accuracy, ties broken by lower NLL.
    Fit on the validation/calibration split only, never on test."""
    y = np.asarray(y, dtype=np.int64)
    n = int(round(1.0 / step))
    best_key: tuple[float, float] | None = None
    best_w: tuple[float, ...] = ()
    for comb in _simplex_grid(len(probs), n):
        w = tuple(c / n for c in comb)
        F = fuse(probs, w)
        key = (-float((F.argmax(1) == y).mean()), nll(F, y))
        if best_key is None or key < best_key:
            best_key, best_w = key, w
    return best_w
