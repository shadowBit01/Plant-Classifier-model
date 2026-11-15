"""Lane L3 (core-ml). Fit the core state for one dataset and write the test ablation.

Usage: python scripts/fit_core.py --dataset vl600 [--device cpu]

Inputs
  $VANALENS_CACHE/emb/<dataset>_{gallery,cal,test}.npz   (EmbeddingSet; gallery X_aug/y_aug used if present)
  $VANALENS_DATA/<dataset>/labels.json                    (taxa, idx order)
  $VANALENS_CACHE/text/<dataset>.npz                      (optional text prototypes, scripts/build_traits.py)
Outputs
  $VANALENS_CACHE/state/<dataset>.json   weights, T, qhat, open-set thr, scorer hyper-params, ensemble block
  $VANALENS_CACHE/state/<dataset>_probe_{linear,mlp}.npz   trained heads (HeadScorer.save, numpy inference)
  runs/L3/ablation_<dataset>.json        test metrics for knn, probe, text, fused, fused+T, fused+T+sets, mlp,
                                         ensemble (+ per-row cal top-1, genus / family top-1)

Heads train on gallery X (+ X_aug). Their hyper-parameters and early stopping, the default head (linear unless the
MLP wins on cal by more than MLP_MARGIN top-1), fusion weights, T, qhat, the open-set threshold and whether the
ensemble is kept are all chosen on cal. Test is only evaluated.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import ece, macro_top1, rank_acc, topk_acc  # noqa: E402
from vanalens import conformal, fusion  # noqa: E402
from vanalens.config import ALPHA, CACHE_DIR, DATA_DIR, RUNS_DIR  # noqa: E402
from vanalens.contracts import EmbeddingSet, Taxon, check_proba, load_taxa  # noqa: E402
from vanalens.scorers import HeadScorer, KNNScorer, ProbeScorer, TextScorer  # noqa: E402

KNN_K = 10
KNN_TEMP = 0.05
PROBE_C_GRID = (1.0, 10.0, 100.0, 1000.0)   # sklearn fallback only (no torch), chosen on cal
TEXT_TEMP = 0.01
OPENSET_PASS = 0.95   # fraction of known-species cal queries that must pass the open-set gate
MLP_MARGIN = 0.01     # MLP replaces the linear probe as default only if it beats it on cal by more than this
# Head grids: every combination is trained and the best is chosen on cal top-1 (ties: NLL).
# weight_decay is an explicit L2 penalty sum(W**2) (see HeadScorer).
LINEAR_GRID = {"weight_decay": (0.0, 1e-5, 1e-4, 1e-3), "label_smoothing": (0.0, 0.1), "balanced": (False, True)}
MLP_GRID = {"weight_decay": (0.0, 1e-5, 1e-4), "label_smoothing": (0.0, 0.1), "balanced": (False, True)}
LINEAR_FIXED = {"hidden": 0, "lr": 1e-3, "epochs": 60, "patience": 8, "scale": 10.0}
MLP_FIXED = {"hidden": 1024, "dropout": 0.5, "lr": 1e-3, "epochs": 60, "patience": 8, "scale": 10.0}


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


def _grid(g: dict) -> list[dict]:
    out: list[dict] = [{}]
    for k, vals in g.items():
        out = [d | {k: v} for d in out for v in vals]
    return out


def fit_head(X, y, Xc, yc, C, fixed, grid, device, log) -> tuple[HeadScorer, list[dict]]:
    """Train every grid config with early stopping on cal; return the best head (cal top-1, ties NLL) + table."""
    best, table = None, []
    for cfg in _grid(grid):
        h = HeadScorer(C, device=device, **fixed, **cfg).fit(X, y, Xc, yc)
        Pc = h.predict_proba(Xc)
        key = (-topk_acc(Pc, yc, 1), fusion.nll(Pc, yc))
        table.append(cfg | {"cal_top1": round(-key[0], 4), "cal_nll": round(key[1], 4), "best_epoch": h.best_epoch})
        log(f"  head hidden={fixed.get('hidden', 0)} {cfg} cal top1 {-key[0]:.4f} ep {h.best_epoch}")
        if best is None or key < best[0]:
            best = (key, h)
    return best[1], table


def calibrated(names: list[str], Pcal: dict, Ptest: dict, y_cal: np.ndarray, alpha: float) -> dict:
    """Fit simplex fusion weights, temperature and qhat on cal for scorers `names`; apply them to test."""
    w = fusion.fit_weights([Pcal[n] for n in names], y_cal) if len(names) > 1 else (1.0,)
    F_cal = fusion.fuse([Pcal[n] for n in names], w)
    T = fusion.fit_temperature(F_cal, y_cal)
    FT_cal = fusion.apply_temperature(F_cal, T)
    qhat = conformal.fit_qhat(FT_cal, y_cal, alpha=alpha)
    F_test = fusion.fuse([Ptest[n] for n in names], w)
    return {"names": names, "weights": w, "T": float(T), "qhat": float(qhat), "F_cal": F_cal, "FT_cal": FT_cal,
            "F_test": F_test, "FT_test": fusion.apply_temperature(F_test, T)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--device", default=None, help="torch device for the heads (default: cuda if available)")
    args = ap.parse_args(argv)
    ds = args.dataset

    def log(m: str) -> None:
        print(m, flush=True)

    taxa = load_taxa(DATA_DIR / ds / "labels.json")
    C = len(taxa)
    emb = {s: EmbeddingSet.load(CACHE_DIR / "emb" / f"{ds}_{s}.npz") for s in ("gallery", "cal", "test")}
    for s, e in emb.items():
        assert e.y is not None, f"{s} split must be labelled"
        assert int(e.y.max()) < C, f"{s} labels exceed labels.json ({C} classes)"
    y_cal, y_test = emb["cal"].y.astype(np.int64), emb["test"].y.astype(np.int64)
    Xc, Xt = emb["cal"].X, emb["test"].X

    # Head training set: gallery X plus the augmented views L2 saved (if any).
    d = np.load(CACHE_DIR / "emb" / f"{ds}_gallery.npz", allow_pickle=False)
    X_tr, y_tr = emb["gallery"].X, emb["gallery"].y.astype(np.int64)
    n_aug = 0
    if "X_aug" in d.files and "y_aug" in d.files and d["X_aug"].shape[0] > 0:
        X_tr = np.concatenate([X_tr, d["X_aug"]]).astype(np.float32)
        y_tr = np.concatenate([y_tr, d["y_aug"].astype(np.int64)])
        n_aug = int(d["y_aug"].shape[0])
    del d

    knn = KNNScorer(emb["gallery"], C, k=KNN_K, temperature=KNN_TEMP)
    state_dir = CACHE_DIR / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    heads: dict[str, object] = {}
    head_info: dict[str, dict] = {}
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False
    if has_torch:
        log(f"{ds}: training heads on {len(y_tr)} rows ({n_aug} augmented), device={args.device or 'auto'}")
        for name, fixed, grid in (("probe", LINEAR_FIXED, LINEAR_GRID), ("mlp", MLP_FIXED, MLP_GRID)):
            h, table = fit_head(X_tr, y_tr, Xc, y_cal, C, fixed, grid, args.device, log)
            path = state_dir / f"{ds}_probe_{'linear' if name == 'probe' else 'mlp'}.npz"
            h.save(path)
            heads[name] = h
            head_info[name] = {"backend": "torch", "weights": str(path), "config": h.config(), "grid": table}
    else:  # sklearn CPU fallback: linear probe only, C chosen on cal
        best = None
        for c_reg in PROBE_C_GRID:
            cand = ProbeScorer(C, C=c_reg).fit(EmbeddingSet([""] * len(y_tr), X_tr, y_tr))
            Pc = cand.predict_proba(Xc)
            key = (-topk_acc(Pc, y_cal, 1), fusion.nll(Pc, y_cal))
            if best is None or key < best[0]:
                best = (key, cand)
        heads["probe"] = best[1]
        head_info["probe"] = {"backend": "sklearn", "C": best[1].C}

    scorers: dict[str, object] = {"knn": knn, **heads}
    text_path = CACHE_DIR / "text" / f"{ds}.npz"
    tp = load_text_protos(text_path, taxa)
    text_missing: list[str] = []
    if tp is not None:
        protos, strs, text_missing = tp
        scorers["text"] = TextScorer(protos, strs, temperature=TEXT_TEMP)

    P = {split: {n: sc.predict_proba(X) for n, sc in scorers.items()} for split, X in (("cal", Xc), ("test", Xt))}
    for split in P:
        for Pm in P[split].values():
            check_proba(Pm, C)
    cal_top1 = {n: topk_acc(P["cal"][n], y_cal, 1) for n in scorers}

    # Default head: the linear probe unless the MLP beats it on cal by more than MLP_MARGIN.
    default_head = "probe"
    if "mlp" in scorers and cal_top1["mlp"] > cal_top1["probe"] + MLP_MARGIN:
        default_head = "mlp"
    Pd_cal = {"knn": P["cal"]["knn"], "probe": P["cal"][default_head]}
    Pd_test = {"knn": P["test"]["knn"], "probe": P["test"][default_head]}
    if "text" in scorers:
        Pd_cal["text"], Pd_test["text"] = P["cal"]["text"], P["test"]["text"]
    base = calibrated(list(Pd_cal), Pd_cal, Pd_test, y_cal, args.alpha)
    base_cal_top1 = topk_acc(base["FT_cal"], y_cal, 1)

    ens = None
    if "mlp" in scorers:
        ens = calibrated([n for n in ("knn", "probe", "mlp", "text") if n in scorers], P["cal"], P["test"],
                         y_cal, args.alpha)
        ens["cal_top1"] = topk_acc(ens["FT_cal"], y_cal, 1)
        ens["kept"] = bool(ens["cal_top1"] > base_cal_top1)

    sim_cal = knn.max_similarity(Xc)
    thr = float(np.quantile(sim_cal, 1.0 - OPENSET_PASS))
    probe_state = {k: v for k, v in head_info[default_head].items() if k != "grid"}
    probe_state["head"] = "linear" if default_head == "probe" else "mlp"
    probe_state.setdefault("C", 100.0)   # only for consumers that still refit the sklearn probe from C
    state = {
        "dataset": ds,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_classes": C,
        "scorers": base["names"],
        "weights": {n: float(w) for n, w in zip(base["names"], base["weights"])},
        "T": base["T"],
        "alpha": float(args.alpha),
        "qhat": base["qhat"],
        "thr": thr,
        "knn": {"k": KNN_K, "temperature": KNN_TEMP},
        "probe": probe_state,
        "default_head": {"choice": probe_state["head"], "rule": f"mlp only if cal top1 > linear + {MLP_MARGIN}",
                         "cal_top1": {n: round(cal_top1[n], 4) for n in heads}},
        "heads": {n: {k: v for k, v in i.items() if k != "grid"} for n, i in head_info.items()},
        "ensemble": (None if ens is None else {
            "scorers": ens["names"], "weights": {n: float(w) for n, w in zip(ens["names"], ens["weights"])},
            "T": ens["T"], "qhat": ens["qhat"], "cal_top1": round(ens["cal_top1"], 4),
            "fused_cal_top1": round(base_cal_top1, 4), "kept": ens["kept"]}),
        "final": "ensemble" if ens is not None and ens["kept"] else "fused",
        "text": ({"temperature": TEXT_TEMP, "protos": str(text_path), "missing_species": text_missing}
                 if "text" in scorers else None),
        "train_rows": {"gallery": len(emb["gallery"].ids), "aug": n_aug},
        "n": {s: len(e.ids) for s, e in emb.items()},
    }
    (state_dir / f"{ds}.json").write_text(json.dumps(state, indent=1))

    # Evaluate on test. Rows without their own conformal step get a qhat fitted on their cal probabilities.
    def row(P_cal: np.ndarray, P_test: np.ndarray, qhat: float | None = None) -> dict[str, float]:
        q = qhat if qhat is not None else conformal.fit_qhat(P_cal, y_cal, alpha=args.alpha)
        m = metrics(P_test, y_test, conformal.predict_sets(P_test, q))
        m["genus_top1"] = round(rank_acc(P_test, y_test, taxa, "genus"), 4)
        m["family_top1"] = round(rank_acc(P_test, y_test, taxa, "family"), 4)
        m["cal_top1"] = round(topk_acc(P_cal, y_cal, 1), 4)
        return m

    rows = {n: row(P["cal"][n], P["test"][n]) for n in ("knn", "probe", "text") if n in scorers}
    rows["fused"] = row(base["F_cal"], base["F_test"])
    rows["fused+T"] = row(base["FT_cal"], base["FT_test"])
    rows["fused+T+sets"] = row(base["FT_cal"], base["FT_test"], base["qhat"])
    rows["fused+T+sets"]["unknown_rate"] = round(float(conformal.is_unknown(knn.max_similarity(Xt), thr).mean()), 4)
    if "mlp" in scorers:
        rows["mlp"] = row(P["cal"]["mlp"], P["test"]["mlp"])
    if ens is not None:
        rows["ensemble"] = row(ens["FT_cal"], ens["FT_test"], ens["qhat"])

    final = "ensemble" if state["final"] == "ensemble" else "fused+T+sets"
    out = {"dataset": ds, "split": "test", "n_test": len(y_test), "final_row": final,
           "head_grids": {n: i.get("grid") for n, i in head_info.items()}, "state": state, "rows": rows}
    out_path = RUNS_DIR / "L3" / f"ablation_{ds}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))

    cols = ["cal_top1", "top1", "top5", "macro_top1", "genus_top1", "family_top1", "nll", "ece", "coverage",
            "mean_set_size"]
    log(f"{ds} test n={len(y_test)} | default head={probe_state['head']} weights={state['weights']} "
        f"T={base['T']:.3f} qhat={base['qhat']:.3f} thr={thr:.3f} | final={final}")
    if ens is not None:
        log(f"ensemble weights={state['ensemble']['weights']} cal top1 {ens['cal_top1']:.4f} vs fused "
            f"{base_cal_top1:.4f} -> kept={ens['kept']}")
    log(f"{'row':<14}" + "".join(f"{c[:11]:>12}" for c in cols))
    for r, m in rows.items():
        log(f"{r:<14}" + "".join(f"{m[c]:>12.4f}" if c in m else f"{'-':>12}" for c in cols))
    if text_missing:
        log(f"text protos missing for {len(text_missing)} species (neutral proto used): {text_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
