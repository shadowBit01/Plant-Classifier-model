"""Final results table: baselines B0-B4 and the VanaLens ablation, per dataset with a test manifest.

Writes runs/L6/final.json and prints the table as markdown (nothing else).

Rows per dataset
- B0 PlantNet API (runs/L1/b0_<ds>.json), B1 BioCLIP zero-shot (runs/L2/b1_<ds>.json),
  B2 (runs/L2/b2_<ds>.json), B3 = linear probe (ablation "probe" row), B4 VLM (runs/L4/b4_<ds>.json).
- Ablation rows from runs/L3/ablation_<ds>.json: knn, probe, text, fused, fused+T, fused+T+sets.
- "+ geo prior" and "+ reranker" computed here with VanaLens.predict on the test manifest. The photos are fed
  through a cached-embedding encoder (the frozen-encoder embeddings L2 extracted, the same ones every other
  row uses), so the step differs from fused+T+sets only by the prior / reranker. No organ gate (no text
  encoder). Reranker only if VANALENS_LLM_KEY is set, capped at MAX_RERANK calls.
Missing results become rows with status "not run". Coverage / set size exist only for rows with sets.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vanalens import config, rerank  # noqa: E402
from vanalens.contracts import EmbeddingSet  # noqa: E402
from vanalens.data import read_manifest  # noqa: E402
from vanalens.pipeline import VanaLens  # noqa: E402

DATASETS = ("inat_india", "dimpsar", "campus")
ABLATION = ("knn", "probe", "text", "fused", "fused+T", "fused+T+sets")
BASELINES = (("B0 PlantNet API", "L1", "b0"), ("B1 BioCLIP zero-shot", "L2", "b1"),
             ("B2 BioCLIP 2 kNN", "L2", "b2"), ("B4 VLM", "L4", "b4"))
MAX_RERANK = 200
COLS = ("n", "top1", "top5", "macro_top1", "coverage", "mean_set_size")


class CachedEncoder:
    """Stands in for the live encoder: returns the cached embedding of each manifest path."""

    def __init__(self, paths: dict[str, int], X: np.ndarray):
        self.paths, self.X = paths, X

    def embed_paths(self, paths: list[str]) -> np.ndarray:
        return self.X[[self.paths[p] for p in paths]]


def _num(v) -> float | None:
    return None if v is None else round(float(v), 4)


def _row(name: str, src: str, d: dict | None, n: int | None = None, status: str = "ok", note: str = "") -> dict:
    d = d or {}
    return {"method": name, "source": src, "status": status, "note": note,
            "n": d.get("n", n) if status == "ok" else None,
            "top1": _num(d.get("top1")), "top5": _num(d.get("top5")), "macro_top1": _num(d.get("macro_top1")),
            "coverage": _num(d.get("coverage")), "mean_set_size": _num(d.get("mean_set_size"))}


def _not_run(name: str, src: str, note: str) -> dict:
    return _row(name, src, None, status="not run", note=note)


def _metrics(P: np.ndarray, y: np.ndarray, top1: np.ndarray, sets: list[list[int]]) -> dict:
    top5 = np.argsort(-P, axis=1, kind="stable")[:, :5]
    correct = top1 == y
    return {"n": int(len(y)), "top1": float(correct.mean()),
            "top5": float(np.mean([y[i] in top5[i] for i in range(len(y))])),
            "macro_top1": float(np.mean([correct[y == c].mean() for c in np.unique(y)])),
            "coverage": float(np.mean([y[i] in sets[i] for i in range(len(y))])),
            "mean_set_size": float(np.mean([len(s) for s in sets]))}


def pipeline_rows(ds: str) -> list[dict]:
    """+ geo prior and + reranker via VanaLens.predict over the cached test embeddings."""
    cache = config.CACHE_DIR
    test = EmbeddingSet.load(cache / "emb" / f"{ds}_test.npz")
    row_of = {i: k for k, i in enumerate(test.ids)}
    samples = [s for s in read_manifest(cache / "manifests" / f"{ds}_test.jsonl")
               if s.label is not None and s.image_id in row_of]
    lens = VanaLens.from_cache(ds)
    lens._encoder = CachedEncoder({s.path: row_of[s.image_id] for s in samples}, test.X)
    y = np.array([s.label for s in samples], dtype=np.int64)
    src = f"VanaLens.predict, cached emb, beta={lens.beta}"

    def run(beta: float):
        lens.beta = beta
        preds = [lens.predict([s.path], lat=s.lat, lon=s.lon, use_rerank=False) for s in samples]
        P = np.vstack([p.probs for p in preds])
        return preds, P

    beta = lens.beta
    preds0, P0 = run(0.0)                      # reproduces fused+T+sets (sanity check)
    check = _metrics(P0, y, P0.argmax(1), [p.pred_set for p in preds0])
    preds, P = run(beta)
    sets = [p.pred_set for p in preds]
    top1 = P.argmax(1)
    n_coords = sum(s.lat is not None and s.lon is not None for s in samples)
    rows = [_row("fused+T+sets (rerun)", f"VanaLens.predict, cached emb, beta=0", check,
                 note="current cache (text protos rebuilt after L3 ablation); baseline for the geo step"),
            _row("+ geo prior", src, _metrics(P, y, top1, sets),
                 note=f"{n_coords}/{len(samples)} photos with coords")]

    if not os.environ.get(config.LLM_KEY_ENV):
        rows.append(_not_run("+ reranker", "VanaLens.predict", f"{config.LLM_KEY_ENV} not set"))
        return rows
    name_to_idx = {t.species: t.idx for t in lens.taxa}
    rr_top1, calls, changed = top1.copy(), 0, 0
    for k, (s, p) in enumerate(zip(samples, preds)):
        if calls >= MAX_RERANK:
            break
        if p.rank == "unknown" and p.rationale or not rerank.should_rerank(p.pred_set):
            continue
        calls += 1
        q = lens.predict([s.path], lat=s.lat, lon=s.lon, use_rerank=True)
        if q.reranked and q.label in name_to_idx:
            changed += int(name_to_idx[q.label] != rr_top1[k])
            rr_top1[k] = name_to_idx[q.label]
    rows.append(_row("+ reranker", "VanaLens.predict + VLM rerank", _metrics(P, y, rr_top1, sets),
                     note=f"{calls} reranker calls (cap {MAX_RERANK}), {changed} top-1 changed"))
    return rows


def dataset_rows(ds: str) -> list[dict]:
    runs = config.RUNS_DIR
    rows: list[dict] = []
    abl_path = runs / "L3" / f"ablation_{ds}.json"
    abl = json.loads(abl_path.read_text()) if abl_path.exists() else None
    n_abl = abl.get("n_test") if abl else None
    for name, lane, key in BASELINES:
        f = runs / lane / f"{key}_{ds}.json"
        d = json.loads(f.read_text()) if f.exists() else None
        if d is None:
            rows.append(_not_run(name, f"runs/{lane}/{key}_{ds}.json", "no result file"))
        elif d.get("status") == "skipped" or not d.get("n"):
            rows.append(_not_run(name, f"runs/{lane}/{key}_{ds}.json", d.get("reason", "skipped")))
        else:
            rows.append(_row(name, f"runs/{lane}/{key}_{ds}.json", {**d, "coverage": None, "mean_set_size": None}))
        if key == "b2":                        # B3 sits between B2 and B4
            if abl and "probe" in abl["rows"]:
                rows.append(_row("B3 linear probe", f"runs/L3/ablation_{ds}.json#probe",
                                 {**abl["rows"]["probe"], "coverage": None, "mean_set_size": None}, n=n_abl))
            else:
                rows.append(_not_run("B3 linear probe", f"runs/L3/ablation_{ds}.json", "no ablation file"))
    for step in ABLATION:
        if abl and step in abl["rows"]:
            r = abl["rows"][step]
            if step != "fused+T+sets":
                r = {**r, "coverage": None, "mean_set_size": None}
            rows.append(_row(step, f"runs/L3/ablation_{ds}.json#{step}", r, n=n_abl))
        else:
            rows.append(_not_run(step, f"runs/L3/ablation_{ds}.json", "no ablation row"))
    try:
        rows += pipeline_rows(ds)
    except (FileNotFoundError, KeyError) as e:
        rows.append(_not_run("+ geo prior", "VanaLens.predict", f"{type(e).__name__}: {e}"))
        rows.append(_not_run("+ reranker", "VanaLens.predict", "geo step failed"))
    return rows


def to_markdown(table: dict) -> str:
    out = []
    for ds, rows in table["datasets"].items():
        out += [f"### {ds}", "", "| method | n | top-1 | top-5 | macro top-1 | coverage | mean set size | note |",
                "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            cells = ["not run" if r["status"] != "ok" else ("-" if r[c] is None else
                     (str(r[c]) if c == "n" else f"{r[c]:.4f}")) for c in COLS]
            out.append(f"| {r['method']} | " + " | ".join(cells) + f" | {r['note']} |")
        out.append("")
    if table["skipped_datasets"]:
        out.append("No test manifest: " + ", ".join(table["skipped_datasets"]))
    return "\n".join(out).rstrip()


def main() -> int:
    present = [d for d in DATASETS if (config.CACHE_DIR / "manifests" / f"{d}_test.jsonl").exists()]
    if "inat_india" not in present:
        present.insert(0, "inat_india")
    table = {"alpha": config.ALPHA, "columns": list(COLS),
             "datasets": {ds: dataset_rows(ds) for ds in present},
             "skipped_datasets": [d for d in DATASETS if d not in present]}
    out = config.RUNS_DIR / "L6" / "final.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    print(to_markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
