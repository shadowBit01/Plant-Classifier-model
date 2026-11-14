"""Lane L3 (core-ml). Combine scorer outputs and calibrate. Tests: tests/test_core.py"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def fuse(probs: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Log-linear pooling: softmax(sum_i w_i * log(P_i + 1e-9)) row-wise. Weights need not sum to 1.
    A weight of 0 removes that scorer. Returns (N, C), rows sum to 1."""
    raise NotImplementedError


def nll(P: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class (clip P at 1e-12)."""
    raise NotImplementedError


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    """softmax(log(P + 1e-12) / T) row-wise. T > 1 softens, T < 1 sharpens."""
    raise NotImplementedError


def fit_temperature(P: np.ndarray, y: np.ndarray, grid: Sequence[float] | None = None) -> float:
    """Return the T (default grid: 48 log-spaced values in [0.2, 5]) minimising nll(apply_temperature(P, T), y)."""
    raise NotImplementedError


def fit_weights(probs: Sequence[np.ndarray], y: np.ndarray, step: float = 0.1) -> tuple[float, ...]:
    """Grid-search weights on the simplex (step `step`) maximising top-1 accuracy, ties broken by lower NLL.
    Fit on the validation/calibration split only, never on test."""
    raise NotImplementedError
