"""Lane L1 (data-eval). One function every lane calls to score a run; writes runs/<lane>/<name>.json.

Usage (CLI): python -m eval.harness --probs runs/x.npz --name b2_bioclip_dimpsar --lane L2
where the npz holds P (N, C), y (N,), optional sets (object array) and labels.json path.
"""
from __future__ import annotations

import numpy as np

from vanalens.contracts import Taxon


def evaluate(P: np.ndarray, y: np.ndarray, taxa: list[Taxon], sets: list[np.ndarray] | None = None) -> dict:
    """Return {"n", "top1", "top5", "macro_top1", "ece", "genus_top1", "family_top1"}
    plus {"coverage", "mean_set_size"} when `sets` is given. Floats rounded to 4 dp."""
    raise NotImplementedError


def save_result(result: dict, name: str, lane: str) -> str:
    """Write RUNS_DIR/<lane>/<name>.json (add a UTC timestamp), return the path."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
