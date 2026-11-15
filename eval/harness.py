"""Lane L1 (data-eval). One function every lane calls to score a run; writes runs/<lane>/<name>.json.

Usage (CLI): python -m eval.harness --probs runs/x.npz --name b2_bioclip_dimpsar --lane L2
where the npz holds P (N, C), y (N,), optional sets (object array) and labels.json path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from vanalens.config import RUNS_DIR
from vanalens.contracts import Taxon, load_taxa
from .metrics import ece, macro_top1, rank_acc, topk_acc


def evaluate(P: np.ndarray, y: np.ndarray, taxa: list[Taxon], sets: list[np.ndarray] | None = None) -> dict:
    """Return {"n", "top1", "top5", "macro_top1", "ece", "genus_top1", "family_top1"}
    plus {"coverage", "mean_set_size"} when `sets` is given. Floats rounded to 4 dp."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=np.int64)
    out: dict = {
        "n": int(len(y)),
        "top1": round(topk_acc(P, y, k=1), 4),
        "top5": round(topk_acc(P, y, k=5), 4),
        "macro_top1": round(macro_top1(P, y), 4),
        "ece": round(ece(P, y), 4),
        "genus_top1": round(rank_acc(P, y, taxa, "genus"), 4),
        "family_top1": round(rank_acc(P, y, taxa, "family"), 4),
    }
    if sets is not None:
        n = len(y)
        if n > 0:
            covered = sum(1 for yi, s in zip(y, sets) if int(yi) in np.asarray(s).tolist())
            mean_size = float(np.mean([len(s) for s in sets]))
            out["coverage"] = round(covered / n, 4)
            out["mean_set_size"] = round(mean_size, 4)
        else:
            out["coverage"] = 0.0
            out["mean_set_size"] = 0.0
    return out


def save_result(result: dict, name: str, lane: str) -> str:
    """Write RUNS_DIR/<lane>/<name>.json (add a UTC timestamp), return the path."""
    out_dir = Path(RUNS_DIR) / lane
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name[:-5] if name.endswith(".json") else name
    out_path = out_dir / f"{stem}.json"
    payload = dict(result)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score predictions and write runs/<lane>/<name>.json")
    parser.add_argument("--probs", required=True, help="Path to .npz with P, y, optional sets and labels path")
    parser.add_argument("--name", required=True, help="Run name")
    parser.add_argument("--lane", required=True, help="Lane ID, e.g. L2")
    parser.add_argument("--labels", default=None, help="Optional override path to labels.json")
    args = parser.parse_args()

    d = np.load(args.probs, allow_pickle=True)
    P = d["P"]
    y = d["y"]
    sets = list(d["sets"]) if "sets" in d else None

    labels_path = args.labels
    if labels_path is None:
        for key in ("labels", "labels_path", "taxa_path"):
            if key in d:
                val = d[key]
                labels_path = str(val.item() if val.ndim == 0 else val[0])
                break
    if labels_path is None:
        raise ValueError("labels.json path must be present in the .npz or passed via --labels")

    taxa = load_taxa(labels_path)
    res = evaluate(P, y, taxa, sets=sets)
    save_result(res, args.name, args.lane)


if __name__ == "__main__":
    main()

