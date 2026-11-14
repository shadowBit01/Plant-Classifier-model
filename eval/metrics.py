"""Lane L1 (data-eval). Metrics. Tests: tests/test_eval.py"""
from __future__ import annotations

import numpy as np

from vanalens.contracts import Taxon


def topk_acc(P: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Fraction of rows whose true class is among the k highest probabilities."""
    raise NotImplementedError


def macro_top1(P: np.ndarray, y: np.ndarray) -> float:
    """Mean over classes present in y of per-class top-1 accuracy."""
    raise NotImplementedError


def ece(P: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins on max prob:
    sum_b (|b|/N) * |acc_b - conf_b|."""
    raise NotImplementedError


def rank_acc(P: np.ndarray, y: np.ndarray, taxa: list[Taxon], rank: str) -> float:
    """Top-1 accuracy after summing probabilities to `rank` ("genus" | "family")."""
    raise NotImplementedError


def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Probability a random positive scores higher than a random negative (ties count 0.5)."""
    raise NotImplementedError
