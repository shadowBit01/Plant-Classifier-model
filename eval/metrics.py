"""Lane L1 (data-eval). Metrics. Tests: tests/test_eval.py"""
from __future__ import annotations

import numpy as np

from vanalens.contracts import Taxon


def topk_acc(P: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Fraction of rows whose true class is among the k highest probabilities."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=np.int64)
    if P.shape[0] == 0:
        return 0.0
    k = max(1, min(int(k), P.shape[1]))
    top_k = np.argsort(-P, axis=1, kind="stable")[:, :k]
    return float(np.any(top_k == y[:, None], axis=1).mean())


def macro_top1(P: np.ndarray, y: np.ndarray) -> float:
    """Mean over classes present in y of per-class top-1 accuracy."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=np.int64)
    classes = np.unique(y)
    if classes.size == 0:
        return 0.0
    preds = P.argmax(axis=1)
    return float(np.mean([np.mean(preds[y == c] == c) for c in classes]))


def ece(P: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins on max prob:
    sum_b (|b|/N) * |acc_b - conf_b|."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    if n == 0:
        return 0.0
    conf = P.max(axis=1)
    acc = (P.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf >= lo) & (conf <= hi if b == n_bins - 1 else conf < hi)
        if np.any(mask):
            total += (np.sum(mask) / n) * abs(float(np.mean(acc[mask])) - float(np.mean(conf[mask])))
    return float(total)


def rank_acc(P: np.ndarray, y: np.ndarray, taxa: list[Taxon], rank: str) -> float:
    """Top-1 accuracy after summing probabilities to `rank` ("genus" | "family")."""
    if rank not in ("genus", "family"):
        raise ValueError(f"rank must be 'genus' or 'family', got {rank!r}")
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=np.int64)
    if P.shape[0] == 0:
        return 0.0
    names = [getattr(t, rank) for t in taxa]
    uniq = list(dict.fromkeys(names))
    name_to_idx = {name: i for i, name in enumerate(uniq)}
    col_to_rank = np.array([name_to_idx[name] for name in names], dtype=np.int64)
    P_rank = np.zeros((P.shape[0], len(uniq)), dtype=float)
    for r in range(len(uniq)):
        P_rank[:, r] = P[:, col_to_rank == r].sum(axis=1)
    y_rank = col_to_rank[y]
    return float(np.mean(P_rank.argmax(axis=1) == y_rank))


def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Probability a random positive scores higher than a random negative (ties count 0.5)."""
    pos = np.asarray(pos_scores, dtype=float).ravel()
    neg = np.asarray(neg_scores, dtype=float).ravel()
    if pos.size == 0 or neg.size == 0:
        return 0.0
    neg_sorted = np.sort(neg)
    less = np.searchsorted(neg_sorted, pos, side="left")
    le = np.searchsorted(neg_sorted, pos, side="right")
    ties = le - less
    return float(np.mean((less + 0.5 * ties) / neg_sorted.size))

