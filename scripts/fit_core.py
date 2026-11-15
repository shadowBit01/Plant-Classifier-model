"""Lane L3 (core-ml). Fit the core state for one dataset and write the test ablation.

Usage: python scripts/fit_core.py --dataset inat_india

Inputs
  $VANALENS_CACHE/emb/<dataset>_{gallery,cal,test}.npz   (EmbeddingSet)
  $VANALENS_DATA/<dataset>/labels.json                    (taxa, idx order)
  $VANALENS_CACHE/text/<dataset>.npz                      (optional text prototypes, scripts/build_traits.py)
Outputs
  $VANALENS_CACHE/state/<dataset>.json   weights, T, qhat, open-set thr, scorer hyper-params
  runs/L3/ablation_<dataset>.json        test metrics for knn, probe, text, fused, fused+T, fused+T+sets

Everything is fitted on gallery (probe) and cal (weights, T, qhat, thr); test is only evaluated.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import ece, macro_top1, topk_acc  # noqa: E402
from vanalens import conformal, fusion  # noqa: E402
from vanalens.config import ALPHA, CACHE_DIR, DATA_DIR, RUNS_DIR  # noqa: E402
from vanalens.contracts import EmbeddingSet, Taxon, check_proba, load_taxa  # noqa: E402
from vanalens.scorers import KNNScorer, ProbeScorer, TextScorer  # noqa: E402

KNN_K = 10
KNN_TEMP = 0.05
PROBE_C_GRID = (1.0, 10.0, 100.0, 1000.0)   # chosen on cal by top-1, ties by NLL
TEXT_TEMP = 0.01
OPENSET_PASS = 0.95   # fraction of known-species cal queries that must pass the open-set gate


def load_text_protos(path: Path, taxa: list[Taxon]) -> tuple[list[np.ndarray], list[list[str]], list[str]] | None:
    """Read build_traits.py output without pickle, aligned to `taxa` by species name.

    Classes without a prototype (protos built before the class was added) get the mean of all other
    prototypes, a neutral "generic plant" direction, and the name prompt string. Returns
    (protos, strings, missing_species) or None if the file does not exist.
    """
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    counts = d["counts"].astype(int)
    flat = d["stacked_protos"].reshape(-1, d["stacked_protos"].shape[-1]).astype(np.float64)
    strings = json.loads(str(d["strings_json"]))
    offs = np.concatenate([[0], np.cumsum(counts)])
    by_name = {str(s): (flat[offs[i]:offs[i + 1]], strings[i]) for i, s in enumerate(d["species"])}
    generic = flat.mean(0, keepdims=True)
    protos, strs, missing = [], [], []
    for t in taxa:
        if t.species in by_name:
            p, s = by_name[t.species]
        else:
            p, s = generic, [f"a photo of {t.species}"]
            missing.append(t.species)
        protos.append(p)
        strs.append(list(s))
    return protos, strs, missing


def metrics(P: np.ndarray, y: np.ndarray, sets: list[np.ndarray] | None = None) -> dict[str, float]:
    out = {
        "top1": topk_acc(P, y, 1),
        "top5": topk_acc(P, y, 5),
        "macro_top1": macro_top1(P, y),
        "nll": fusion.nll(P, y),
        "ece": ece(P, y),
    }
    if sets is not None:
        out["coverage"] = conformal.coverage(sets, y)
        out["mean_set_size"] = conformal.mean_set_size(sets)
        out["singleton_rate"] = float(np.mean([len(s) == 1 for s in sets]))
    return {k: round(float(v), 4) for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args(argv)
    ds = args.dataset

    taxa = load_taxa(DATA_DIR / ds / "labels.json")
    C = len(taxa)
    emb = {s: EmbeddingSet.load(CACHE_DIR / "emb" / f"{ds}_{s}.npz") for s in ("gallery", "cal", "test")}
    for s, e in emb.items():
        assert e.y is not None, f"{s} split must be labelled"
        assert int(e.y.max()) < C, f"{s} labels exceed labels.json ({C} classes)"
    y_cal, y_test = emb["cal"].y.astype(np.int64), emb["test"].y.astype(np.int64)

    knn = KNNScorer(emb["gallery"], C, k=KNN_K, temperature=KNN_TEMP)
    best = None
    for c_reg in PROBE_C_GRID:
        cand = ProbeScorer(C, C=c_reg).fit(emb["gallery"])
        Pc = cand.predict_proba(emb["cal"].X)
        key = (-topk_acc(Pc, y_cal, 1), fusion.nll(Pc, y_cal))
        if best is None or key < best[0]:
            best = (key, cand)
    probe = best[1]
    scorers: dict[str, object] = {"knn": knn, "probe": probe}
    text_path = CACHE_DIR / "text" / f"{ds}.npz"
    tp = load_text_protos(text_path, taxa)
    text_missing: list[str] = []
    if tp is not None:
        protos, strs, text_missing = tp
        scorers["text"] = TextScorer(protos, strs, temperature=TEXT_TEMP)
    names = list(scorers)

    P = {split: {n: sc.predict_proba(emb[split].X) for n, sc in scorers.items()} for split in ("cal", "test")}
    for split in P:
        for Pm in P[split].values():
            check_proba(Pm, C)

    # Fit on cal only.
    weights = fusion.fit_weights([P["cal"][n] for n in names], y_cal)
    F_cal = fusion.fuse([P["cal"][n] for n in names], weights)
    T = fusion.fit_temperature(F_cal, y_cal)
    FT_cal = fusion.apply_temperature(F_cal, T)
    qhat = conformal.fit_qhat(FT_cal, y_cal, alpha=args.alpha)
    sim_cal = knn.max_similarity(emb["cal"].X)
    thr = float(np.quantile(sim_cal, 1.0 - OPENSET_PASS))

    state = {
        "dataset": ds,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_classes": C,
        "scorers": names,
        "weights": {n: float(w) for n, w in zip(names, weights)},
        "T": float(T),
        "alpha": float(args.alpha),
        "qhat": float(qhat),
        "thr": thr,
        "knn": {"k": KNN_K, "temperature": KNN_TEMP},
        "probe": {"C": probe.C},
        "text": ({"temperature": TEXT_TEMP, "protos": str(text_path), "missing_species": text_missing}
                 if "text" in scorers else None),
        "n": {s: len(e.ids) for s, e in emb.items()},
    }
    state_path = CACHE_DIR / "state" / f"{ds}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1))

    # Evaluate on test.
    F_test = fusion.fuse([P["test"][n] for n in names], weights)
    FT_test = fusion.apply_temperature(F_test, T)
    sets = conformal.predict_sets(FT_test, qhat)
    rows = {n: metrics(P["test"][n], y_test) for n in names}
    rows["fused"] = metrics(F_test, y_test)
    rows["fused+T"] = metrics(FT_test, y_test)
    rows["fused+T+sets"] = metrics(FT_test, y_test, sets)
    unknown_rate = float(conformal.is_unknown(knn.max_similarity(emb["test"].X), thr).mean())
    rows["fused+T+sets"]["unknown_rate"] = round(unknown_rate, 4)

    out = {"dataset": ds, "split": "test", "n_test": len(y_test), "state": state, "rows": rows}
    out_path = RUNS_DIR / "L3" / f"ablation_{ds}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))

    cols = ["top1", "top5", "macro_top1", "nll", "ece", "coverage", "mean_set_size"]
    print(f"{ds} test n={len(y_test)} | weights={state['weights']} T={T:.3f} qhat={qhat:.3f} thr={thr:.3f}")
    print(f"{'row':<14}" + "".join(f"{c:>14}" for c in cols))
    for r, m in rows.items():
        print(f"{r:<14}" + "".join(f"{m[c]:>14.4f}" if c in m else f"{'-':>14}" for c in cols))
    if text_missing:
        print(f"text protos missing for {len(text_missing)} species (neutral proto used): {text_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
