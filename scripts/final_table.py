"""Final results table (L6-c): every number with a 95% bootstrap CI, per dataset with a test manifest.

Writes runs/L6/final.json and prints the tables as markdown (nothing else).

Datasets: inat_india, vl600, campus (each only if CACHE_DIR/manifests/<ds>_test.jsonl exists).
All VanaLens numbers come from the cached frozen-encoder embeddings (CPU), through the same code the live
pipeline runs (VanaLens.scores -> VanaLens.combine -> conformal / backoff). Nothing is tuned on test: fusion
weights, T and qhat come from L3's state (cal), beta from L6's cal fit, the selective threshold from cal.

CIs: 95% percentile bootstrap, N_BOOT resamples of test OBSERVATIONS (Sample.group), seed 0. One resample
matrix per dataset is shared by every metric, so all CIs of a dataset use the same resamples.

Sections per dataset (final.json: datasets[ds] = list of rows, extras[ds] = the other sections)
- rows: baselines (point values from runs/<lane>/<key>_<ds>.json; no per-sample predictions there, so no CI),
  ablation recomputed from the cache with CIs (L3's reported value in the note as a cross-check),
  "VanaLens 1 photo (no geo)", "+ geo prior" (1 photo), "VanaLens up to 3 photos + geo" (per observation),
  "+ reranker" (only with VANALENS_LLM_KEY). Columns: COLS.
- selective: threshold on max probability (1 photo, with geo) = the smallest t whose answered cal photos reach
  >= TARGET accuracy; test accuracy on answered photos and the answered share.
- open_set: hold out HOLDOUT of the species (seed 0); on the rest refit a linear head (state's linear-head
  config, CPU, early stop on known cal), T, qhat and the open-set threshold (5% cal quantile of max gallery
  cosine, as L3); fusion = state's fused config (knn/probe/text weights; the MLP is not refit: too slow on CPU).
  Reject rate on held-out species, false-reject rate on known species, AUROC (max gallery cosine, known = +).
- few_shot: on that refit model, add SHOTS gallery photos of every held-out species via add_species; top-1
  (over all classes) on their test photos, and known-species top-1 afterwards.
- target: which headline numbers reach TARGET (point value, and whether the CI lower bound does too).

`--baselines [ds ...]` (L6-e, default vl600): adds final.json["baselines"][ds] without touching any other key:
B2 zero-shot (L2 point values, CI not available), BioCLIP-only / DINO-only linear probes (refit on gallery X + X_aug,
early stop on cal), kNN only, the main linear head alone, text prototypes only, and the full VanaLens rows copied
from the main table. top-1 / top-5 / macro top-1 with the main table's observation bootstrap.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vanalens import config, conformal, fusion, hierarchy, rerank  # noqa: E402
from vanalens.config import BACKOFF_TAU  # noqa: E402
from vanalens.contracts import EmbeddingSet, Taxon, load_taxa  # noqa: E402
from vanalens.data import read_manifest  # noqa: E402
from vanalens.pipeline import MAX_PHOTOS, VanaLens  # noqa: E402
from vanalens.scorers import HeadScorer  # noqa: E402

DATASETS = ("inat_india", "vl600", "campus")
BASELINES = (("B0 PlantNet API", "L1", "b0"), ("B1 BioCLIP zero-shot", "L2", "b1"),
             ("B2 BioCLIP 2 kNN", "L2", "b2"), ("B4 VLM", "L4", "b4"))
ABLATION = ("knn", "probe", "mlp", "text", "fused", "fused+T", "fused+T+sets", "ensemble")
COLS = ("top1", "top5", "macro_top1", "genus_top1", "family_top1", "backoff_acc", "species_share",
        "coverage", "mean_set_size")
N_BOOT = 1000
SEED = 0
TARGET = 0.95
HOLDOUT = 0.10
SHOTS = (1, 3, 5, 10)
OPENSET_PASS = 0.95
MAX_RERANK = 200


class CachedEncoder:
    """Stands in for the live encoder: returns the cached embedding of each known path. No organ gate (no
    encode_text) unless `text` is given."""

    def __init__(self, paths: dict[str, int], X: np.ndarray):
        self.paths, self.X = paths, X

    def embed_paths(self, paths: list[str]) -> np.ndarray:
        return self.X[[self.paths[p] for p in paths]]


class CachedTextEncoder(CachedEncoder):
    """Few-shot eval only: add_species asks the BioCLIP 2 text tower for "a photo of <species>". Without loading
    the tower on CPU, answer with the mean of that species' cached text prototypes (CACHE_DIR/text/<ds>.npz: BioCLIP 2
    text embeddings of "a photo of <species>, which has <trait>"): same modality and the same species-name prompt,
    no images. (The pipeline's own no-text-tower fallback, an image centroid, is on a different cosine scale.)"""

    def __init__(self, paths: dict[str, int], X: np.ndarray, protos: dict[str, np.ndarray]):
        super().__init__(paths, X)
        self.protos = protos

    def encode_text(self, strings: list[str]) -> np.ndarray:
        out = []
        for s in strings:
            v = self.protos[s.removeprefix("a photo of ")].mean(0)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float32)


# ------------------------------------------------------------------------------------------- bootstrap
class Boot:
    """Observation-level bootstrap. Units (photos or observations) carry a group id; every statistic is a ratio of
    per-unit sums, so one (N_BOOT, G) matrix of group multiplicities serves all of them."""

    def __init__(self, groups: list[str], n_boot: int = N_BOOT, seed: int = SEED):
        self.keys = sorted(set(groups))
        self.pos = {g: i for i, g in enumerate(self.keys)}
        G = len(self.keys)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, G, size=(n_boot, G))
        self.W = np.stack([np.bincount(r, minlength=G) for r in idx]).astype(np.float64)

    def gidx(self, groups: list[str]) -> np.ndarray:
        return np.array([self.pos[g] for g in groups], dtype=np.int64)

    def _g(self, gi: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.bincount(gi, weights=np.asarray(v, dtype=np.float64), minlength=len(self.keys))

    def ratio(self, gi: np.ndarray, num, den=None) -> dict:
        num = np.asarray(num, dtype=np.float64)
        den = np.ones_like(num) if den is None else np.asarray(den, dtype=np.float64)
        if den.sum() == 0:
            return {"value": None, "ci": None}
        ng, dg = self._g(gi, num), self._g(gi, den)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = (self.W @ ng) / (self.W @ dg)
        return _stat(num.sum() / den.sum(), b)

    def macro(self, gi: np.ndarray, correct, y) -> dict:
        y = np.asarray(y, dtype=np.int64)
        cls, yi = np.unique(y, return_inverse=True)
        G = len(self.keys)
        num = np.zeros((G, len(cls)))
        den = np.zeros((G, len(cls)))
        np.add.at(num, (gi, yi), np.asarray(correct, dtype=np.float64))
        np.add.at(den, (gi, yi), 1.0)
        point = float(np.mean(num.sum(0) / den.sum(0)))
        with np.errstate(invalid="ignore", divide="ignore"):
            per = (self.W @ num) / (self.W @ den)
        return _stat(point, np.nanmean(per, axis=1))

    def auroc(self, gi: np.ndarray, score, pos) -> dict:
        score = np.asarray(score, dtype=np.float64)
        pos = np.asarray(pos, dtype=bool)
        _, inv = np.unique(score, return_inverse=True)

        def auc(w: np.ndarray) -> float:
            wp = np.bincount(inv, weights=w * pos)
            wn = np.bincount(inv, weights=w * ~pos)
            tot = wp.sum() * wn.sum()
            return float(np.sum(wp * (np.cumsum(wn) - 0.5 * wn)) / tot) if tot > 0 else np.nan

        return _stat(auc(np.ones(len(score))), np.array([auc(w[gi]) for w in self.W]))


def _stat(point: float, boots: np.ndarray) -> dict:
    boots = boots[np.isfinite(boots)]
    ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)] \
        if boots.size else None
    return {"value": round(float(point), 4), "ci": ci}


# ------------------------------------------------------------------------------------------- metrics
def backoff_arrays(P: np.ndarray, taxa: list[Taxon], y: np.ndarray, tau: float = BACKOFF_TAU):
    """Vectorised hierarchy.backoff: (answered, correct-at-reported-rank, at-species-rank) per row."""
    Pg, gn = hierarchy.aggregate(P, taxa, "genus")
    Pf, fn = hierarchy.aggregate(P, taxa, "family")
    j, g, f = P.argmax(1), Pg.argmax(1), Pf.argmax(1)
    sp = P[np.arange(len(P)), j] >= tau
    ge = ~sp & (Pg[np.arange(len(P)), g] >= tau)
    fa = ~sp & ~ge & (Pf[np.arange(len(P)), f] >= tau)
    tg = np.array([gn.index(taxa[c].genus) for c in y])
    tf = np.array([fn.index(taxa[c].family) for c in y])
    correct = (sp & (j == y)) | (ge & (g == tg)) | (fa & (f == tf))
    return sp | ge | fa, correct, sp


def cls_metrics(P: np.ndarray, y: np.ndarray, gi: np.ndarray, boot: Boot, taxa: list[Taxon],
                qhat: float | None) -> dict:
    """COLS with CIs for probability rows P (one row per unit) and labels y."""
    order = np.argsort(-P, axis=1, kind="stable")
    top1 = order[:, 0] == y
    top5 = (order[:, :5] == y[:, None]).any(1)
    genus = np.array([taxa[a].genus == taxa[b].genus for a, b in zip(order[:, 0], y)])
    family = np.array([taxa[a].family == taxa[b].family for a, b in zip(order[:, 0], y)])
    answered, bcorrect, sp = backoff_arrays(P, taxa, y)
    out = {"n": int(len(y)), "top1": boot.ratio(gi, top1), "top5": boot.ratio(gi, top5),
           "macro_top1": boot.macro(gi, top1, y), "genus_top1": boot.ratio(gi, genus),
           "family_top1": boot.ratio(gi, family), "backoff_acc": boot.ratio(gi, bcorrect, answered),
           "species_share": boot.ratio(gi, sp)}
    if qhat is not None:
        sets = conformal.predict_sets(P, qhat)
        out["coverage"] = boot.ratio(gi, [int(t) in s.tolist() for s, t in zip(sets, y)])
        out["mean_set_size"] = boot.ratio(gi, [len(s) for s in sets])
    return out


def selective_threshold(conf: np.ndarray, correct: np.ndarray, target: float = TARGET) -> float:
    """Smallest t such that accuracy over {conf >= t} >= target (answers as many as possible); inf if none."""
    order = np.argsort(-conf, kind="stable")
    c, ok = conf[order], correct[order].astype(np.float64)
    acc = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    last = np.r_[c[1:] != c[:-1], True]          # only cut at the end of a run of tied confidences
    good = np.where(last & (acc >= target))[0]
    return float(c[good[-1]]) if good.size else float("inf")


# ------------------------------------------------------------------------------------------- rows
def _row(name: str, src: str, m: dict | None, status: str = "ok", note: str = "") -> dict:
    m = m or {}
    r = {"method": name, "source": src, "status": status, "note": note,
         "n": m.get("n") if status == "ok" else None}
    ci = {}
    for c in COLS:
        v = m.get(c)
        if isinstance(v, dict):
            r[c], ci[c] = v["value"], v["ci"]
        else:
            r[c] = None if v is None else round(float(v), 4)
    r["ci"] = ci
    return r


def _not_run(name: str, src: str, note: str) -> dict:
    return _row(name, src, None, status="not run", note=note)


def load_split(ds: str, split: str):
    cache = config.CACHE_DIR
    emb = EmbeddingSet.load(cache / "emb" / f"{ds}_{split}.npz")
    by_id = {s.image_id: s for s in read_manifest(cache / "manifests" / f"{ds}_{split}.jsonl")}
    keep = [k for k, i in enumerate(emb.ids) if i in by_id and by_id[i].label is not None]
    samples = [by_id[emb.ids[k]] for k in keep]
    y = np.array([s.label for s in samples], dtype=np.int64)
    groups = [s.group or s.image_id for s in samples]
    return emb.X[keep], y, samples, groups


def photo_probs(lens: VanaLens, FT: np.ndarray, samples) -> np.ndarray:
    """Single-photo final distributions (geo prior applied when lens.beta > 0)."""
    return np.vstack([lens.combine(FT[i:i + 1], s.lat, s.lon) for i, s in enumerate(samples)])


def obs_probs(lens: VanaLens, FT: np.ndarray, samples, groups):
    """Per observation: first MAX_PHOTOS photos (manifest order) combined by VanaLens.combine."""
    rows: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        rows.setdefault(g, []).append(i)
    keys = list(rows)
    P = np.vstack([lens.combine(FT[rows[g][:MAX_PHOTOS]], samples[rows[g][0]].lat, samples[rows[g][0]].lon)
                   for g in keys])
    y = np.array([samples[rows[g][0]].label for g in keys], dtype=np.int64)
    n_photos = np.array([min(len(rows[g]), MAX_PHOTOS) for g in keys])
    return P, y, keys, n_photos


def ablation_rows(ds: str, lens: VanaLens, Xt: np.ndarray, yt: np.ndarray, gi, boot, taxa) -> list[dict]:
    """knn / probe / mlp / text / fused / fused+T / fused+T+sets / ensemble from the cache, with CIs."""
    st = lens.state
    abl_path = config.RUNS_DIR / "L3" / f"ablation_{ds}.json"
    l3 = json.loads(abl_path.read_text())["rows"] if abl_path.exists() else {}
    comp = {"knn": lens.knn.predict_proba(Xt)}
    heads = st.get("heads") or {}
    for n in ("probe", "mlp"):                   # rows: linear head and MLP head
        path = (heads.get(n) or {}).get("weights") or (st.get("probe", {}).get("weights") if n == "probe" else None)
        if path and Path(path).exists():
            comp[n] = HeadScorer.load(path).predict_proba(Xt)
    if lens.text is not None:
        comp["text"] = lens.text.predict_proba(Xt)
    default = st.get("probe", {}).get("weights")  # the fused config's "probe" is L3's default head
    fcomp = dict(comp, probe=HeadScorer.load(default).predict_proba(Xt))         if default and Path(default).exists() else comp
    names = [n for n in st["scorers"] if n in fcomp]
    fused = fusion.fuse([fcomp[n] for n in names], [float(st["weights"].get(n, 0)) for n in names])
    comp["fused"] = fused
    comp["fused+T"] = fusion.apply_temperature(fused, float(st["T"]))
    comp["fused+T+sets"] = comp["fused+T"]
    ens = st.get("ensemble") or {}
    if ens:
        en = [n for n in ens["scorers"] if n in comp]
        comp["ensemble"] = fusion.apply_temperature(
            fusion.fuse([comp[n] for n in en], [float(ens["weights"].get(n, 0)) for n in en]), float(ens["T"]))
    rows = []
    for step in ABLATION:
        src = f"cache recompute; runs/L3/ablation_{ds}.json#{step}"
        if step not in comp:
            rows.append(_not_run(step, src, "component missing in cache/state"))
            continue
        qhat = float(st["qhat"]) if step == "fused+T+sets" else float(ens["qhat"]) if step == "ensemble" else None
        m = cls_metrics(comp[step], yt, gi, boot, taxa, qhat)
        ref = l3.get(step, {}).get("top1")
        note = f"L3 top-1 {ref}" if ref is not None else "not in L3 ablation"
        if step == "ensemble":
            note += f"; final config: {lens.config_name}"
        rows.append(_row(step, src, m, note=note))
    return rows


def baseline_rows(ds: str, abl_rows: list[dict]) -> list[dict]:
    rows = []
    for name, lane, key in BASELINES:
        f = config.RUNS_DIR / lane / f"{key}_{ds}.json"
        src = f"runs/{lane}/{key}_{ds}.json"
        d = json.loads(f.read_text()) if f.exists() else None
        if d is None:
            rows.append(_not_run(name, src, "no result file"))
        elif d.get("status") == "skipped" or not d.get("n"):
            rows.append(_not_run(name, src, d.get("reason", "skipped")))
        else:
            r = _row(name, src, {k: d.get(k) for k in ("top1", "top5", "macro_top1", "genus_top1", "family_top1")})
            r["n"] = d.get("n")
            r["note"] = "point value from the result file (no per-sample predictions: no CI)"
            rows.append(r)
        if key == "b2":                          # B3 = linear probe, recomputed with CI
            probe = next((r for r in abl_rows if r["method"] == "probe" and r["status"] == "ok"), None)
            rows.append({**probe, "method": "B3 linear probe"} if probe else
                        _not_run("B3 linear probe", f"runs/L3/ablation_{ds}.json", "no probe head"))
    return rows


# ------------------------------------------------------------------------------------------- open set / few-shot
def _subset_text(lens: VanaLens, keep: list[int]):
    if lens.text is None:
        return None
    return [lens.text.protos[c] for c in keep], [lens.text.trait_strings[c] for c in keep]


def open_set_and_few_shot(ds: str, lens: VanaLens, taxa: list[Taxon], Xt, yt, gt, boot: Boot) -> tuple[dict, list]:
    cache = config.CACHE_DIR
    st = lens.state
    C = len(taxa)
    rng = np.random.default_rng(SEED)
    held = sorted(rng.choice(C, size=max(1, int(round(HOLDOUT * C))), replace=False).tolist())
    held_set = set(held)
    known = [c for c in range(C) if c not in held_set]
    remap = np.full(C, -1, dtype=np.int64)
    remap[known] = np.arange(len(known))
    ktaxa = [replace(taxa[c], idx=i) for i, c in enumerate(known)]

    gal = EmbeddingSet.load(cache / "emb" / f"{ds}_gallery.npz")
    gk = np.isin(gal.y, known)
    kgal = EmbeddingSet([i for i, m in zip(gal.ids, gk) if m], gal.X[gk], remap[gal.y[gk]])
    Xc, yc, _, _ = load_split(ds, "cal")
    ck = np.isin(yc, known)

    # refit the linear head on known species (gallery + augmented views), CPU, early stop on known cal
    d = np.load(cache / "emb" / f"{ds}_gallery.npz", allow_pickle=False)
    X_tr, y_tr = gal.X[gk], remap[gal.y[gk]]
    if "X_aug" in d.files and "y_aug" in d.files and d["X_aug"].shape[0] > 0:
        ak = np.isin(d["y_aug"], known)
        X_tr = np.concatenate([X_tr, d["X_aug"][ak]]).astype(np.float32)
        y_tr = np.concatenate([y_tr, remap[d["y_aug"][ak].astype(np.int64)]])
    hcfg = dict((st.get("heads") or {}).get("probe", {}).get("config") or st.get("probe", {}).get("config") or {})
    hcfg.pop("best_epoch", None)
    head = HeadScorer(len(known), device="cpu", **hcfg).fit(X_tr, y_tr, Xc[ck], remap[yc[ck]])
    out_dir = cache / "state" / "openset"
    out_dir.mkdir(parents=True, exist_ok=True)
    head_path = out_dir / f"{ds}_probe_linear_known.npz"
    head.save(head_path)

    st_os = {k: v for k, v in st.items() if k not in ("final", "ensemble", "beta")}
    st_os["probe"] = {**st.get("probe", {}), "weights": str(head_path)}
    st_os["n_classes"] = len(known)
    text = _subset_text(lens, known)
    os_lens = VanaLens(ktaxa, kgal, st_os, text=text, encoder=CachedEncoder({}, Xt))
    T = fusion.fit_temperature(_raw(os_lens, Xc[ck]), remap[yc[ck]])
    os_lens.T = T
    qhat = conformal.fit_qhat(os_lens.scores(Xc[ck]), remap[yc[ck]], config.ALPHA)
    os_lens.qhat = qhat
    thr = float(np.quantile(os_lens.knn.max_similarity(Xc[ck]), 1.0 - OPENSET_PASS))
    os_lens.thr = thr
    st_os.update(T=T, qhat=qhat, thr=thr)

    sim = os_lens.knn.max_similarity(Xt)
    is_known = np.isin(yt, known)
    reject = sim < thr
    ki = np.where(is_known)[0]
    Pk = os_lens.scores(Xt[ki])
    k_top1 = np.zeros(len(yt))
    k_top1[ki] = Pk.argmax(1) == remap[yt[ki]]
    open_set = {
        "held_out_species": len(held), "known_species": len(known), "seed": SEED,
        "n_test_held_out": int((~is_known).sum()), "n_test_known": int(is_known.sum()),
        "refit": {"head": "linear (state config, CPU)", "T": round(T, 4), "qhat": round(qhat, 4),
                  "thr": round(thr, 4), "fusion": dict(zip(os_lens.names, os_lens.weights))},
        "reject_rate_held_out": boot.ratio(gt, reject & ~is_known, ~is_known),
        "false_reject_rate_known": boot.ratio(gt, reject & is_known, is_known),
        "auroc": boot.auroc(gt, sim, is_known),
        "known_top1_refit": boot.ratio(gt, k_top1, is_known),
    }

    # few-shot: add k gallery photos of every held-out species (nested: first k of one permutation per species)
    perm = {c: rng.permutation(np.where(gal.y == c)[0]) for c in held}
    held_protos = {taxa[c].species: np.atleast_2d(lens.text.protos[c]) for c in held} if lens.text is not None else {}
    hi = np.where(~is_known)[0]
    curve = []
    for k in SHOTS:
        enc = CachedTextEncoder({f"gallery/{gal.ids[r]}": int(r) for c in held for r in perm[c][:k]}, gal.X,
                                held_protos)
        fl = VanaLens(ktaxa, kgal, st_os, text=_subset_text(lens, known), encoder=enc)
        new_idx: dict[int, int | None] = {}
        for c in held:
            ps = [f"gallery/{gal.ids[r]}" for r in perm[c][:k]]
            if ps:
                fl.add_species(taxa[c], ps)
            new_idx[c] = len(fl.taxa) - 1 if ps else None
        target = np.array([new_idx[c] if new_idx[c] is not None else -1 for c in yt[hi]])
        P = fl.scores(Xt)
        pred = P.argmax(1)
        top1_new = np.zeros(len(yt))
        top1_new[hi] = pred[hi] == target
        top1_known = np.zeros(len(yt))
        top1_known[ki] = pred[ki] == remap[yt[ki]]
        curve.append({"shots": k, "species_added": int(sum(v is not None for v in new_idx.values())),
                      "top1_new": boot.ratio(gt, top1_new, ~is_known),
                      "top1_known_after": boot.ratio(gt, top1_known, is_known)})
    return open_set, curve


def _raw(lens: VanaLens, X: np.ndarray) -> np.ndarray:
    """Fused probabilities before temperature (T = 1) for refitting T."""
    T, lens.T = lens.T, 1.0
    try:
        return lens.scores(X)
    finally:
        lens.T = T


# ------------------------------------------------------------------------------------------- dataset
def reranker_row(lens: VanaLens, samples, P1: np.ndarray, y: np.ndarray, gi, boot, taxa) -> dict:
    if not os.environ.get(config.LLM_KEY_ENV):
        return _not_run("+ reranker", "VanaLens.predict", f"{config.LLM_KEY_ENV} not set")
    name_to_idx = {t.species: t.idx for t in lens.taxa}
    sets = conformal.predict_sets(P1, lens.qhat)
    top1, calls = P1.argmax(1).copy(), 0
    for k, s in enumerate(samples):
        if calls >= MAX_RERANK:
            break
        if not rerank.should_rerank([int(i) for i in sets[k]]):
            continue
        calls += 1
        q = lens.predict([s.path], lat=s.lat, lon=s.lon, use_rerank=True)
        if q.reranked and q.label in name_to_idx:
            top1[k] = name_to_idx[q.label]
    m = {"n": len(y), "top1": boot.ratio(gi, top1 == y)}
    return _row("+ reranker", "VanaLens.predict + VLM rerank", m, note=f"{calls} calls (cap {MAX_RERANK})")


def dataset_table(ds: str) -> tuple[list[dict], dict]:
    cache = config.CACHE_DIR
    taxa = load_taxa(config.DATA_DIR / ds / "labels.json")
    Xt, yt, st_samples, groups = load_split(ds, "test")
    boot = Boot(groups)
    gi = boot.gidx(groups)
    lens = VanaLens.from_cache(ds)
    lens._encoder = CachedEncoder({s.path: i for i, s in enumerate(st_samples)}, Xt)

    abl = ablation_rows(ds, lens, Xt, yt, gi, boot, taxa)
    rows = baseline_rows(ds, abl) + abl

    FT = lens.scores(Xt)
    beta = lens.beta
    n_coords = sum(s.lat is not None and s.lon is not None for s in st_samples)
    lens.beta = 0.0
    P0 = photo_probs(lens, FT, st_samples)
    lens.beta = beta
    P1 = photo_probs(lens, FT, st_samples)
    Po, yo, okeys, n_ph = obs_probs(lens, FT, st_samples, groups)
    go = boot.gidx(okeys)
    # sanity: the full predict path equals combine on the first few observations
    for s in st_samples[:3]:
        assert np.allclose(lens.predict([s.path], s.lat, s.lon, use_rerank=False).probs,
                           lens.combine(FT[[st_samples.index(s)]], s.lat, s.lon), atol=1e-8)
    src = f"VanaLens.combine on cached emb ({lens.config_name}, T={lens.T:.3f}, qhat={lens.qhat:.3f})"
    rows.append(_row("VanaLens 1 photo (no geo)", src, cls_metrics(P0, yt, gi, boot, taxa, lens.qhat),
                     note="per photo"))
    rows.append(_row("+ geo prior", src + f", beta={beta}", cls_metrics(P1, yt, gi, boot, taxa, lens.qhat),
                     note=f"per photo; {n_coords}/{len(yt)} photos with coords"))
    multi = int((n_ph > 1).sum())
    rows.append(_row("VanaLens up to 3 photos + geo", src + f", beta={beta}",
                     cls_metrics(Po, yo, go, boot, taxa, lens.qhat),
                     note=f"per observation; {multi}/{len(yo)} observations have 2-3 photos, "
                          f"mean {n_ph.mean():.2f} photos"))
    rows.append(reranker_row(lens, st_samples, P1, yt, gi, boot, taxa))

    # selective accuracy: threshold on cal (1 photo, with geo), applied to test
    Xc, yc, c_samples, _ = load_split(ds, "cal")
    Pc = photo_probs(lens, lens.scores(Xc), c_samples)
    t = selective_threshold(Pc.max(1), Pc.argmax(1) == yc)
    ans = P1.max(1) >= t
    ok = P1.argmax(1) == yt
    ans_c = Pc.max(1) >= t
    selective = {"threshold": None if not np.isfinite(t) else round(t, 4), "target": TARGET,
                 "cal_accuracy_answered": round(float((Pc.argmax(1) == yc)[ans_c].mean()), 4) if ans_c.any() else None,
                 "cal_share_answered": round(float(ans_c.mean()), 4),
                 "test_accuracy_answered": boot.ratio(gi, ok & ans, ans),
                 "test_share_answered": boot.ratio(gi, ans)}

    open_set, few = open_set_and_few_shot(ds, lens, taxa, Xt, yt, gi, boot)

    by = {r["method"]: r for r in rows}

    def tgt(name: str, row: str, col: str) -> dict:
        r = by[row]
        v, ci = r.get(col), r.get("ci", {}).get(col)
        return {"metric": name, "value": v, "ci": ci, "reaches_95": v is not None and v >= TARGET,
                "ci_low_reaches_95": bool(ci and ci[0] >= TARGET)}

    target = [tgt("single-photo top-1", "+ geo prior", "top1"),
              tgt("3-photo top-1", "VanaLens up to 3 photos + geo", "top1"),
              tgt("single-photo top-5", "+ geo prior", "top5"),
              tgt("3-photo top-5", "VanaLens up to 3 photos + geo", "top5"),
              tgt("genus top-1 (1 photo)", "+ geo prior", "genus_top1"),
              tgt("family top-1 (1 photo)", "+ geo prior", "family_top1"),
              tgt("backoff accuracy at reported rank (1 photo)", "+ geo prior", "backoff_acc")]
    sa = selective["test_accuracy_answered"]
    target.append({"metric": "selective accuracy (answered photos)", "value": sa["value"], "ci": sa["ci"],
                   "reaches_95": sa["value"] is not None and sa["value"] >= TARGET,
                   "ci_low_reaches_95": bool(sa["ci"] and sa["ci"][0] >= TARGET),
                   "share_answered": selective["test_share_answered"]})
    extras = {"n_test_photos": len(yt), "n_test_observations": len(boot.keys), "beta": beta,
              "config": lens.config_name, "selective": selective, "open_set": open_set, "few_shot": few,
              "target": target}
    return rows, extras


# ------------------------------------------------------------------------------------------- baselines (L6-e)
BCOLS = ("top1", "top5", "macro_top1")
FULL_ROWS = (("VanaLens full, 1 photo (no geo)", "VanaLens 1 photo (no geo)"),
             ("VanaLens full, 1 photo + geo", "+ geo prior"),
             ("VanaLens full, up to 3 photos + geo", "VanaLens up to 3 photos + geo"))


def _brow(name: str, src: str, m: dict, note: str = "") -> dict:
    """Baselines-block row: top-1, top-5, macro top-1, each with its 95% CI (by observation)."""
    return {"method": name, "source": src, "n": m["n"], **{c: m[c]["value"] for c in BCOLS},
            "ci": {c: m[c]["ci"] for c in BCOLS}, "note": note}


def _bmetrics(P: np.ndarray, y: np.ndarray, gi: np.ndarray, boot: Boot) -> dict:
    order = np.argsort(-P, axis=1, kind="stable")
    top1 = order[:, 0] == y
    return {"n": int(len(y)), "top1": boot.ratio(gi, top1),
            "top5": boot.ratio(gi, (order[:, :5] == y[:, None]).any(1)), "macro_top1": boot.macro(gi, top1, y)}


def _block_probe(X_tr, y_tr, X_cal, y_cal, n_classes: int, cfg: dict, device: str | None) -> HeadScorer:
    """Linear head on one feature block (or the concatenation): the main linear head's config (balanced loss,
    scale, lr, epochs), early stop on cal. dino_weight is 1: a single block needs no block weight."""
    cfg = {k: v for k, v in cfg.items() if k not in ("best_epoch", "dino_weight", "hidden", "dropout")}
    return HeadScorer(n_classes, hidden=0, device=device, dino_weight=1.0, **cfg).fit(X_tr, y_tr, X_cal, y_cal)


def compute_baselines(data: dict, state: dict, text, n_classes: int, boot: Boot, b2: dict | None = None,
                      main_rows: list[dict] | None = None, device: str | None = None) -> list[dict]:
    """Head-to-head baselines on one test split, all from cached frozen embeddings.

    data: Xg, yg, ids (gallery = kNN index and head training rows), Xa, ya (augmented gallery views, may be
    empty), Xc, yc (cal: early stopping only), Xt, yt, gt (test rows and their bootstrap group index).
    state: L3 state (knn params, the linear head's config and saved weights); text: (protos, strings) or None.
    Nothing reads test except the final scoring.
    """
    from vanalens.config import BIOCLIP_DIM
    from vanalens.scorers import KNNScorer, TextScorer

    Xt, yt, gt = data["Xt"], data["yt"], data["gt"]
    X_tr = np.concatenate([data["Xg"], data["Xa"]]).astype(np.float32) if len(data["Xa"]) else data["Xg"]
    y_tr = np.concatenate([data["yg"], data["ya"]]).astype(np.int64) if len(data["Xa"]) else data["yg"]
    aug = f"gallery {len(data['Xg'])} + aug {len(data['Xa'])} rows"
    rows = []
    if b2 and b2.get("n"):
        rows.append({"method": "B2 BioCLIP 2 zero-shot", "source": "runs/L2/b2_<ds>.json", "n": b2["n"],
                     **{c: b2.get(c) for c in BCOLS}, "ci": {c: "not available" for c in BCOLS},
                     "note": "point values from L2 (no per-photo predictions saved: CI not available)"})
    else:
        rows.append({"method": "B2 BioCLIP 2 zero-shot", "source": "runs/L2/b2_<ds>.json", "n": None,
                     **{c: None for c in BCOLS}, "ci": {c: "not available" for c in BCOLS}, "note": "no result"})
    hcfg = dict((state.get("heads") or {}).get("probe", {}).get("config") or state.get("probe", {}).get("config")
                or {})
    blocks = (("BioCLIP 2-only linear probe", slice(0, BIOCLIP_DIM)),
              ("DINO-only linear probe", slice(BIOCLIP_DIM, None)))
    for name, sl in blocks:
        h = _block_probe(X_tr[:, sl], y_tr, data["Xc"][:, sl], data["yc"], n_classes, hcfg, device)
        rows.append(_brow(name, "HeadScorer linear, refit", _bmetrics(h.predict_proba(Xt[:, sl]), yt, gt, boot),
                          note=f"{aug}; early stop on cal (best epoch {h.best_epoch})"))
    kp = state.get("knn", {})
    gal = EmbeddingSet(list(data["ids"]), data["Xg"], data["yg"])
    knn = KNNScorer(gal, n_classes, k=kp.get("k", 10), temperature=kp.get("temperature", 0.05),
                    dino_weight=kp.get("dino_weight", 1.0))
    rows.append(_brow("kNN only (both blocks)", "KNNScorer, state params",
                      _bmetrics(knn.predict_proba(Xt), yt, gt, boot),
                      note=f"gallery {len(data['Xg'])} photos; k={knn.k}, dino_weight={knn.dino_weight} (cal)"))
    path = state.get("probe", {}).get("weights")
    if path and Path(path).exists() and state.get("probe", {}).get("config", {}).get("hidden", 0) == 0:
        h, src = HeadScorer.load(path), "saved L3 linear head (state probe)"
        note = "the main model's linear head alone, no fusion / T / geo"
    else:
        h = HeadScorer(n_classes, hidden=0, device=device,
                       **{k: v for k, v in hcfg.items() if k not in ("best_epoch", "hidden", "dropout")}
                       ).fit(X_tr, y_tr, data["Xc"], data["yc"])
        src, note = "HeadScorer linear, refit", "refit on concatenated features"
    rows.append(_brow("Concatenated-feature linear probe (no fusion)", src,
                      _bmetrics(h.predict_proba(Xt), yt, gt, boot), note=f"{note}; {aug}"))
    if text is not None:
        ts = TextScorer(list(text[0]), [list(s) for s in text[1]],
                        temperature=float(state.get("text", {}).get("temperature", 0.01)))
        rows.append(_brow("Text prototypes only (0 images/species)", "TextScorer, cached BioCLIP 2 text protos",
                          _bmetrics(ts.predict_proba(Xt), yt, gt, boot),
                          note="trait + species-name prompts; no gallery images"))
    by = {r["method"]: r for r in main_rows or []}
    for name, src in FULL_ROWS:
        r = by.get(src)
        if r and r.get("status") == "ok":
            rows.append({"method": name, "source": f"final.json main table: {src}", "n": r["n"],
                         **{c: r[c] for c in BCOLS}, "ci": {c: r["ci"].get(c) for c in BCOLS}, "note": r["note"]})
    return rows


def baselines_for(ds: str, main_rows: list[dict]) -> list[dict]:
    """Load the cached splits of ds and run compute_baselines with the main table's bootstrap (same resamples)."""
    cache = config.CACHE_DIR
    taxa = load_taxa(config.DATA_DIR / ds / "labels.json")
    Xt, yt, _, groups = load_split(ds, "test")
    boot = Boot(groups)
    Xc, yc, _, _ = load_split(ds, "cal")
    d = np.load(cache / "emb" / f"{ds}_gallery.npz", allow_pickle=False)
    gal = EmbeddingSet.load(cache / "emb" / f"{ds}_gallery.npz")
    has_aug = "X_aug" in d.files and "y_aug" in d.files
    data = {"Xg": gal.X, "yg": np.asarray(gal.y, dtype=np.int64), "ids": gal.ids,
            "Xa": d["X_aug"] if has_aug else np.empty((0, gal.X.shape[1]), np.float32),
            "ya": d["y_aug"].astype(np.int64) if has_aug else np.empty(0, np.int64),
            "Xc": Xc, "yc": yc, "Xt": Xt, "yt": yt, "gt": boot.gidx(groups)}
    lens = VanaLens.from_cache(ds)
    text = (lens.text.protos, lens.text.trait_strings) if lens.text is not None else None
    b2f = config.RUNS_DIR / "L2" / f"b2_{ds}.json"
    b2 = json.loads(b2f.read_text()) if b2f.exists() else None
    rows = compute_baselines(data, lens.state, text, len(taxa), boot, b2, main_rows)
    for r in rows:
        r["source"] = r["source"].replace("<ds>", ds)
    return rows


def baselines_markdown(block: dict) -> str:
    out = []
    for ds, rows in block.items():
        out += [f"### baselines: {ds}", "", "| method | n | top-1 | top-5 | macro top-1 | note |",
                "|---|---|---|---|---|---|"]
        for r in rows:
            cells = [_cell(r[c], r["ci"][c] if isinstance(r["ci"].get(c), list) else None) for c in BCOLS]
            out.append(f"| {r['method']} | {r['n']} | " + " | ".join(cells) + f" | {r['note']} |")
        out.append("")
    return "\n".join(out).rstrip()


def main_baselines(datasets: list[str]) -> int:
    """Add the "baselines" block to runs/L6/final.json; every other key is left untouched."""
    path = config.RUNS_DIR / "L6" / "final.json"
    table = json.loads(path.read_text())
    block = dict(table.get("baselines", {}))
    for ds in datasets:
        block[ds] = baselines_for(ds, table["datasets"][ds])
    table["baselines"] = block
    table["baselines_info"] = {
        "columns": list(BCOLS), "ci": f"95% percentile bootstrap over test observations ({N_BOOT}, seed {SEED}), "
        "same resamples as the main table",
        "protocol": "cached frozen embeddings; heads trained on gallery X + X_aug (as the main model), early stop on "
                    "cal; kNN/text params from L3 state (cal); test only scored"}
    path.write_text(json.dumps(table, indent=1))
    print(baselines_markdown({ds: block[ds] for ds in datasets}))
    return 0


# ------------------------------------------------------------------------------------------- output
def _cell(v, ci=None, n=False) -> str:
    if v is None:
        return "-"
    if n:
        return str(v)
    return f"{v:.4f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def _sd(d: dict) -> str:
    return _cell(d.get("value"), d.get("ci"))


def to_markdown(table: dict) -> str:
    out = []
    hdr = ("top-1", "top-5", "macro top-1", "genus top-1", "family top-1", "backoff acc", "species-rank share",
           "coverage", "mean set size")
    for ds, rows in table["datasets"].items():
        ex = table["extras"].get(ds, {})
        out += [f"### {ds}", "", f"95% bootstrap CIs over {ex.get('n_test_observations', '?')} test observations "
                f"({N_BOOT} resamples, seed {SEED}); config {ex.get('config')}, beta {ex.get('beta')}.", "",
                "| method | n | " + " | ".join(hdr) + " | note |", "|---" * (len(hdr) + 3) + "|"]
        for r in rows:
            if r["status"] != "ok":
                cells = ["not run"] * (len(COLS) + 1)
            else:
                cells = [_cell(r["n"], n=True)] + [_cell(r[c], r["ci"].get(c)) for c in COLS]
            out.append(f"| {r['method']} | " + " | ".join(cells) + f" | {r['note']} |")
        if "selective" in ex:
            s = ex["selective"]
            out += ["", f"**Selective** (1 photo, threshold {s['threshold']} on cal for >= {TARGET:.0%}): "
                    f"test accuracy on answered {_sd(s['test_accuracy_answered'])}, "
                    f"answered share {_sd(s['test_share_answered'])}"]
            o = ex["open_set"]
            out += ["", f"**Open set** ({o['held_out_species']} held-out / {o['known_species']} known species, "
                    f"refit thr {o['refit']['thr']}): reject held-out {_sd(o['reject_rate_held_out'])}, "
                    f"false-reject known {_sd(o['false_reject_rate_known'])}, AUROC {_sd(o['auroc'])}, "
                    f"known top-1 {_sd(o['known_top1_refit'])}", "",
                    "| shots per new species | top-1 on new species | known top-1 after |", "|---|---|---|"]
            out += [f"| {f['shots']} | {_sd(f['top1_new'])} | {_sd(f['top1_known_after'])} |" for f in ex["few_shot"]]
            out += ["", "| target >= 95% | value [95% CI] | reaches | CI low reaches |", "|---|---|---|---|"]
            for t in ex["target"]:
                extra = f" (answered {_sd(t['share_answered'])})" if "share_answered" in t else ""
                out.append(f"| {t['metric']} | {_cell(t['value'], t['ci'])}{extra} | "
                           f"{'yes' if t['reaches_95'] else 'no'} | {'yes' if t['ci_low_reaches_95'] else 'no'} |")
        out.append("")
    if table["skipped_datasets"]:
        out.append("No test manifest: " + ", ".join(table["skipped_datasets"]))
    return "\n".join(out).rstrip()


def main() -> int:
    if "--baselines" in sys.argv:            # L6-e: python scripts/final_table.py --baselines [ds ...]
        rest = [a for a in sys.argv[sys.argv.index("--baselines") + 1:] if not a.startswith("-")]
        return main_baselines(rest or ["vl600"])
    present = [d for d in DATASETS if (config.CACHE_DIR / "manifests" / f"{d}_test.jsonl").exists()]
    datasets, extras = {}, {}
    for ds in present:
        datasets[ds], extras[ds] = dataset_table(ds)
    table = {"alpha": config.ALPHA, "target": TARGET, "n_boot": N_BOOT, "seed": SEED, "columns": list(COLS),
             "datasets": datasets, "extras": extras, "skipped_datasets": [d for d in DATASETS if d not in present]}
    out = config.RUNS_DIR / "L6" / "final.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    print(to_markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
