"""Lane L6 (integration) owns VanaLens. MockPipeline is implemented now so the app lane can start at hour 0.

Predict flow (VanaLens.predict):
  1 organ = zero-shot BioCLIP 2 over ORGANS; reject "other" as not-a-plant
  2 X = encoder.embed_paths(paths)                 (one row per photo)
  3 P = fusion.fuse([knn, probe, text], weights)   then fusion.apply_temperature(P, T)
  4 if lat/lon: P = geo.apply_prior(P, geo.geo_prior(gbif_species_counts(...), taxa), beta)
  5 multi-photo (up to MAX_PHOTOS of one plant): p = renormalised exp(mean over photos of log P)
  6 set = conformal.predict_sets(p, qhat); unknown if conformal.is_unknown(max sim, thr)
  7 rank, label, conf = hierarchy.backoff(p, taxa, BACKOFF_TAU)
  8 if rerank.should_rerank(set): rerank over top-5 of set
  9 evidence = knn.neighbors(...); traits = text.explain(...)
Fitted state (weights, T, qhat, thr, beta) is saved to CACHE_DIR/state/<dataset>.json by the integration lane.

Implementation notes (L6-a)
- State (weights, T, qhat, thr, scorer hyper-params) comes from L3-b's state file; L6 adds "beta" (geo prior
  strength, fitted on cal by NLL over BETA_GRID). Missing "beta" -> 0 (no prior).
- Heads (L6-c): the probe is HeadScorer.load(state["probe"]["weights"]) (L3-c torch head, pure-numpy inference);
  only when that file is missing is the sklearn probe refitted on the gallery from state["probe"]["C"].
  If state["final"] == "ensemble" (and the ensemble block is kept), the ensemble scorers / weights / T / qhat are
  used and each head is HeadScorer.load(state["heads"][name]["weights"]) ("probe" = linear, "mlp").
- Text scorer is skipped when CACHE_DIR/text/<dataset>.npz is missing; reranker when VANALENS_LLM_KEY is empty.
- Geo prior is applied ONCE after the multi-photo combination (step 5 before step 4), so k photos of one plant
  do not apply the prior k times. The combination is the MEAN of the calibrated log-probabilities (L6-c), so an
  observation of k photos is exactly as confident as its photos agree, not k times more (keeps the conformal qhat,
  fitted on single photos, meaningful). Photos beyond MAX_PHOTOS are ignored. VanaLens.combine is that step alone,
  for evaluation on cached embeddings.
- add_species: the heads (probe, mlp) cannot grow columns without retraining, so for classes added after fitting it defers to
  the kNN scorer (probe mass scaled by 1 - kNN mass on the new classes). The text scorer gets an
  "a photo of <species>" prototype from the BioCLIP 2 text tower.
- CLI (L6 owns no scripts):  python -m vanalens.pipeline fit-beta --dataset inat_india
                             python -m vanalens.pipeline e2e --dataset inat_india --n 20
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import config, conformal, fusion, geo, hierarchy, rerank
from .config import BACKOFF_TAU, BIOCLIP_DIM, LLM_KEY_ENV
from .contracts import EmbeddingSet, Prediction, Taxon, check_embeddings, load_taxa
from .scorers import HeadScorer, KNNScorer, ProbeScorer, TextScorer

BETA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
ORGAN_PROMPTS = {
    "leaf": "a photo of a plant leaf",
    "flower": "a photo of a flower",
    "fruit": "a photo of a fruit or seed pod",
    "bark": "a photo of tree bark or a stem",
    "whole_plant": "a photo of a whole plant, tree or shrub",
    "other": "a photo of an object, animal, person or building, not a plant",
}
N_EVIDENCE = 3
N_RERANK = 5
MAX_PHOTOS = 3
HEADS = ("probe", "mlp")


def _load_encoder():
    """The real encoder (lazy: torch only loads on first photo). Tests monkeypatch this function.
    VANALENS_DEVICE overrides the device (default cuda; Encoder falls back to cpu when CUDA is absent)."""
    from .encoders import Encoder

    return Encoder(device=os.environ.get("VANALENS_DEVICE", "cuda"))


def _load_text_protos(path: Path, taxa: list[Taxon]):
    """Reuse L3's pickle-free reader (scripts/fit_core.load_text_protos) so class alignment is identical."""
    spec = importlib.util.spec_from_file_location("_vl_fit_core", config.REPO_DIR / "scripts" / "fit_core.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_text_protos(Path(path), taxa)


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _read_paths(manifest: Path) -> dict[str, str]:
    from .data import read_manifest

    return {s.image_id: s.path for s in read_manifest(manifest)} if manifest.exists() else {}


class VanaLens:
    def __init__(self, taxa: list[Taxon], gallery: EmbeddingSet, state: dict,
                 text: tuple[list[np.ndarray], list[list[str]]] | None = None,
                 id_to_path: dict[str, str] | None = None, encoder=None):
        """Build scorers from a labelled gallery and a fitted state dict (format of L3-b's state file)."""
        self.taxa = list(taxa)
        self.state = state
        self.n_fit = len(self.taxa)            # classes the probe knows
        kp = state.get("knn", {})
        self.knn = KNNScorer(gallery, self.n_fit, k=kp.get("k", 10), temperature=kp.get("temperature", 0.05),
                              dino_weight=float(kp.get("dino_weight", 1.0)))
        ens = state.get("ensemble") or {}
        self.config_name = "ensemble" if state.get("final") == "ensemble" and ens.get("kept", True) else "fused"
        cfg = ens if self.config_name == "ensemble" else state
        self.heads: dict = {}
        for n in cfg["scorers"]:
            if n in HEADS and float(cfg["weights"].get(n, 0)) != 0:
                self.heads[n] = self._load_head(n, state, gallery, self.config_name == "ensemble")
        self.probe = self.heads.get("probe")
        self.text: TextScorer | None = None
        if text is not None:
            tt = (state.get("text") or {}).get("temperature", 0.01)
            self.text = TextScorer(list(text[0]), [list(s) for s in text[1]], temperature=tt)
        w = cfg["weights"]
        self.names = [n for n in cfg["scorers"] if (n != "text" or self.text is not None) and w.get(n, 0) != 0]
        if not self.names:                     # only the (missing) text scorer had weight
            self.names = ["probe"]
            self.heads["probe"] = self.probe = self._load_head("probe", state, gallery, False)
        self.weights = [float(w.get(n, 1.0)) for n in self.names]
        self.T = float(cfg["T"])
        self.qhat = float(cfg["qhat"])
        self.thr = float(state["thr"])
        self.beta = float(state.get("beta", 0.0))
        self.id_to_path = dict(id_to_path or {})
        self.class_ids: dict[int, list[str]] = {}
        for i, c in zip(gallery.ids, gallery.y):
            self.class_ids.setdefault(int(c), []).append(i)
        self._encoder = encoder
        self._organ_protos: np.ndarray | None = None
        self._prior_cache: dict[tuple[float, float], np.ndarray | None] = {}

    def _load_head(self, name: str, state: dict, gallery: EmbeddingSet, ensemble: bool):
        """Saved L3-c head (HeadScorer .npz) if its file exists, else (probe only) the sklearn probe refit.
        Fused config: "probe" is state["probe"] (L3's default head, linear or MLP). Ensemble config: each scorer
        name is its own head, state["heads"][name] (so "probe" is the linear head even when the default is MLP)."""
        heads = state.get("heads") or {}
        if name == "probe" and not (ensemble and "probe" in heads):
            info = state.get("probe", {})
        else:
            info = heads.get(name, {})
        path = info.get("weights")
        if path and Path(path).exists():
            head = HeadScorer.load(path)
            if head.n_classes != self.n_fit:
                raise ValueError(f"{name} head has {head.n_classes} classes, labels.json has {self.n_fit}")
            return head
        if name != "probe":
            raise FileNotFoundError(f"state names a {name!r} head but its weights file is missing: {path}")
        return ProbeScorer(self.n_fit, C=info.get("C", 1.0)).fit(gallery)

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_cache(cls, dataset: str) -> "VanaLens":
        """Load state/<dataset>.json, labels.json, the gallery embeddings and (optional) text prototypes."""
        cache = config.CACHE_DIR
        state_path = cache / "state" / f"{dataset}.json"
        if not state_path.exists():
            raise FileNotFoundError(f"{state_path} missing: run scripts/fit_core.py --dataset {dataset} (L3-b)")
        state = json.loads(state_path.read_text())
        taxa = load_taxa(config.DATA_DIR / dataset / "labels.json")
        gallery = EmbeddingSet.load(cache / "emb" / f"{dataset}_gallery.npz")
        text = None
        text_path = cache / "text" / f"{dataset}.npz"
        if "text" in state.get("scorers", []) and text_path.exists():
            tp = _load_text_protos(text_path, taxa)
            if tp is not None:
                text = (tp[0], tp[1])
        id_to_path = _read_paths(cache / "manifests" / f"{dataset}_gallery.jsonl")
        return cls(taxa, gallery, state, text=text, id_to_path=id_to_path)

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = _load_encoder()
        return self._encoder

    # ---------------------------------------------------------------- scoring
    def scores(self, X: np.ndarray) -> np.ndarray:
        """(N, C) fused + temperature-scaled probabilities for embedded photos (no geo prior)."""
        X = np.atleast_2d(X)
        C = len(self.taxa)
        Pk = self.knn.predict_proba(X)
        out = []
        for n in self.names:
            if n == "knn":
                out.append(Pk)
            elif n in self.heads:
                Pp = self.heads[n].predict_proba(X)
                if C > self.n_fit:             # defer to kNN for classes added after fitting
                    new = Pk[:, self.n_fit:]
                    Pp = np.hstack([Pp * (1.0 - new.sum(1, keepdims=True)), new])
                    Pp = Pp / Pp.sum(1, keepdims=True)
                out.append(Pp)
            elif n == "text":
                out.append(self.text.predict_proba(X))
        return fusion.apply_temperature(fusion.fuse(out, self.weights), self.T)

    def _organ(self, X: np.ndarray) -> str | None:
        """Zero-shot organ over ORGANS with BioCLIP 2 text prompts; mean similarity over photos."""
        if not hasattr(self.encoder, "encode_text"):
            return None
        if self._organ_protos is None:
            self._organ_protos = _l2(self.encoder.encode_text([ORGAN_PROMPTS[o] for o in ORGAN_PROMPTS]))
        S = _l2(X[:, :BIOCLIP_DIM]) @ self._organ_protos.T
        return list(ORGAN_PROMPTS)[int(S.mean(0).argmax())]

    def _geo(self, lat: float | None, lon: float | None) -> np.ndarray | None:
        if lat is None or lon is None or self.beta == 0:
            return None
        key = (float(lat), float(lon))
        if key not in self._prior_cache or (self._prior_cache[key] is not None
                                            and len(self._prior_cache[key]) != len(self.taxa)):
            counts = geo.gbif_species_counts(lat, lon)
            self._prior_cache[key] = geo.geo_prior(counts, self.taxa) if counts else None
        return self._prior_cache[key]

    def combine(self, FT: np.ndarray, lat: float | None = None, lon: float | None = None) -> np.ndarray:
        """(k, C) calibrated per-photo probabilities of ONE plant -> (C,) distribution: mean of the log-probs over
        the first MAX_PHOTOS photos, renormalised, then the geo prior (once) when beta > 0 and coords are given."""
        FT = np.atleast_2d(FT)[:MAX_PHOTOS]
        logp = np.log(FT + 1e-12).mean(0)
        p = np.exp(logp - logp.max())
        p /= p.sum()
        prior = self._geo(lat, lon)
        if prior is not None:
            p = geo.apply_prior(p[None], prior, self.beta)[0]
        return p

    def _traits(self, c: int, n: int = 4) -> list[str]:
        return list(self.text.trait_strings[c][:n]) if self.text is not None else []

    # ---------------------------------------------------------------- public API
    def predict(self, paths: list[str], lat: float | None = None, lon: float | None = None,
                use_rerank: bool = True) -> Prediction:
        if not paths:
            raise ValueError("predict needs at least one photo")
        paths = list(paths)[:MAX_PHOTOS]
        X = np.asarray(self.encoder.embed_paths(paths), dtype=np.float32)
        check_embeddings(X)
        organ = self._organ(X)
        p = self.combine(self.scores(X), lat, lon)

        pred_set = [int(i) for i in conformal.predict_sets(p[None], self.qhat)[0]]
        sim = self.knn.max_similarity(X)
        unknown = bool(conformal.is_unknown(np.array([sim.max()]), self.thr)[0])
        rank, label, conf = hierarchy.backoff(p, self.taxa, BACKOFF_TAU)
        order = np.argsort(-p, kind="stable")
        best = int(sim.argmax())
        top = int(order[0])
        evidence = self.knn.neighbors(X[best:best + 1], N_EVIDENCE)[0]
        traits = self.text.explain(X[best:best + 1], top) if self.text is not None else []

        reranked, rationale = False, ""
        if organ == "other":
            rank, label, conf = "unknown", "", float(p[top])
            rationale = "organ gate: the photo does not look like a plant"
        elif unknown:
            rank, label, conf = "unknown", "", float(p[top])
            rationale = f"open-set gate: best gallery similarity {sim.max():.3f} < {self.thr:.3f} (species not in gallery?)"
        elif use_rerank and os.environ.get(LLM_KEY_ENV) and rerank.should_rerank(pred_set):
            cands = [self.taxa[i] for i in pred_set[:N_RERANK]]
            refs = {t.idx: [self.id_to_path[i] for i in self.class_ids.get(t.idx, [])[:2] if i in self.id_to_path]
                    for t in cands}
            chosen, rationale, reranked = rerank.rerank(list(paths), cands, refs,
                                                        {t.species: self._traits(t.idx) for t in cands})
            if reranked and 0 <= chosen < len(self.taxa):
                rank, label, conf = "species", self.taxa[chosen].species, float(p[chosen])
                traits = self.text.explain(X[best:best + 1], chosen) if self.text is not None else traits

        return Prediction(
            probs=p, top_k=[(int(i), float(p[i])) for i in order[:5]], pred_set=pred_set,
            rank=rank, label=label, confidence=float(conf), evidence=evidence, traits=traits,
            organ=organ, reranked=bool(reranked), rationale=rationale,
        )

    def add_species(self, taxon: Taxon, paths: list[str]) -> None:
        """Few-shot: embed `paths`, append to the kNN gallery, add a text prototype. No retraining."""
        if not paths:
            raise ValueError("add_species needs at least one photo")
        idx = len(self.taxa)
        t = replace(taxon, idx=idx)
        X = np.asarray(self.encoder.embed_paths(list(paths)), dtype=np.float32)
        check_embeddings(X)
        ids = [f"added/{t.species}/{k}" for k in range(len(paths))]
        self.taxa.append(t)
        self.knn.n_classes = len(self.taxa)
        self.knn.add(EmbeddingSet(ids, X, np.full(len(ids), idx, dtype=np.int64)))
        self.id_to_path.update(zip(ids, paths))
        self.class_ids[idx] = ids
        if self.text is not None:
            prompt = f"a photo of {t.species}"
            if hasattr(self.encoder, "encode_text"):
                proto = _l2(self.encoder.encode_text([prompt]))
            else:                              # no text tower: image centroid of the BioCLIP block
                proto = _l2(_l2(X[:, :BIOCLIP_DIM]).mean(0))
            self.text.protos.append(proto)
            self.text.trait_strings.append([prompt])


# -------------------------------------------------------------------- fitting / evaluation
def fit_beta(lens: VanaLens, X_cal: np.ndarray, y_cal: np.ndarray,
             coords: list[tuple[float | None, float | None]], grid=BETA_GRID) -> tuple[float, dict[str, float]]:
    """Pick the geo-prior strength beta on cal by NLL. Rows without coords or GBIF counts get a uniform prior."""
    FT = lens.scores(X_cal)
    C = FT.shape[1]
    priors = np.full_like(FT, 1.0 / C)
    for i, (la, lo) in enumerate(coords):
        if la is not None and lo is not None:
            counts = geo.gbif_species_counts(la, lo)
            if counts:
                priors[i] = geo.geo_prior(counts, lens.taxa)
    y = np.asarray(y_cal, dtype=np.int64)
    table = {}
    for b in grid:
        Pb = np.vstack([geo.apply_prior(FT[i:i + 1], priors[i], b) for i in range(len(FT))])
        table[str(b)] = round(fusion.nll(Pb, y), 5)
    beta = float(min(grid, key=lambda b: (table[str(b)], b)))
    return beta, table


def fit_beta_cached(dataset: str) -> dict:
    """Fit beta on the cal split of `dataset` and add it to CACHE_DIR/state/<dataset>.json."""
    from .data import read_manifest

    cache = config.CACHE_DIR
    lens = VanaLens.from_cache(dataset)
    cal = EmbeddingSet.load(cache / "emb" / f"{dataset}_cal.npz")
    by_id = {s.image_id: s for s in read_manifest(cache / "manifests" / f"{dataset}_cal.jsonl")}
    coords = [(by_id[i].lat, by_id[i].lon) if i in by_id else (None, None) for i in cal.ids]
    beta, table = fit_beta(lens, cal.X, cal.y, coords)
    state_path = cache / "state" / f"{dataset}.json"
    state = json.loads(state_path.read_text())
    state["beta"] = beta
    state["beta_fit"] = {"split": "cal", "metric": "nll", "nll": table,
                         "n_with_coords": int(sum(la is not None for la, _ in coords))}
    state_path.write_text(json.dumps(state, indent=1))
    return state["beta_fit"] | {"beta": beta}


def run_e2e(dataset: str, n: int = 20, manifest: Path | None = None, seed: int = 0,
            use_rerank: bool = True) -> dict:
    """Predict `n` labelled observations (all photos of a group, up to MAX_PHOTOS) one by one through the full
    pipeline; save runs/L6/e2e_<dataset>.json."""
    from .data import read_manifest

    manifest = manifest or config.CACHE_DIR / "manifests" / f"{dataset}_test.jsonl"
    obs: dict[str, list] = {}
    for s in read_manifest(manifest):
        if s.label is not None:
            obs.setdefault(s.group or s.image_id, []).append(s)
    keys = sorted(obs)
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(keys), size=min(n, len(keys)), replace=False).tolist())
    t0 = time.perf_counter()
    lens = VanaLens.from_cache(dataset)
    load_s = time.perf_counter() - t0
    rows = []
    for k in pick:
        group = obs[keys[k]][:MAX_PHOTOS]
        s = group[0]
        t1 = time.perf_counter()
        pr = lens.predict([g.path for g in group], lat=s.lat, lon=s.lon, use_rerank=use_rerank)
        sec = time.perf_counter() - t1
        rows.append({"group": keys[k], "n_photos": len(group), "label": int(s.label), "top1": int(pr.top_k[0][0]),
                     "correct": int(pr.top_k[0][0] == s.label), "in_set": int(s.label in pr.pred_set),
                     "set_size": len(pr.pred_set), "rank": pr.rank, "organ": pr.organ,
                     "reranked": pr.reranked, "seconds": round(sec, 3)})
    secs = [r["seconds"] for r in rows]
    out = {
        "dataset": dataset, "manifest": manifest.name, "n": len(rows), "seed": seed,
        "top1": round(float(np.mean([r["correct"] for r in rows])), 4),
        "coverage": round(float(np.mean([r["in_set"] for r in rows])), 4),
        "mean_set_size": round(float(np.mean([r["set_size"] for r in rows])), 4),
        "species_rank_rate": round(float(np.mean([r["rank"] == "species" for r in rows])), 4),
        "unknown_rate": round(float(np.mean([r["rank"] == "unknown" for r in rows])), 4),
        "reranked_rate": round(float(np.mean([r["reranked"] for r in rows])), 4),
        "n_photos": int(sum(r["n_photos"] for r in rows)),
        "seconds_per_obs": round(float(np.mean(secs)), 3),
        "seconds_per_obs_warm": round(float(np.mean(secs[1:])) if len(secs) > 1 else float(secs[0]), 3),
        "load_seconds": round(load_s, 2),
        "device": os.environ.get("VANALENS_DEVICE", "cuda"),
        "config": lens.config_name, "scorers": lens.names, "weights": lens.weights, "beta": lens.beta,
        "rerank_enabled": bool(use_rerank and os.environ.get(LLM_KEY_ENV)),
        "rows": rows,
    }
    out_path = config.RUNS_DIR / "L6" / f"e2e_{manifest.stem.replace('_test', '')}.json"
    out["encoder"] = type(lens.encoder).__name__
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    return out


class MockPipeline:
    """Deterministic fake predictions with the real Prediction shape, for UI work before the model exists."""

    TAXA = [
        Taxon(0, "Ficus religiosa", "Ficus", "Moraceae"),
        Taxon(1, "Ficus benghalensis", "Ficus", "Moraceae"),
        Taxon(2, "Azadirachta indica", "Azadirachta", "Meliaceae"),
        Taxon(3, "Ocimum tenuiflorum", "Ocimum", "Lamiaceae"),
        Taxon(4, "Mangifera indica", "Mangifera", "Anacardiaceae"),
    ]

    def __init__(self):
        self.taxa = list(self.TAXA)

    def predict(self, paths: list[str], lat: float | None = None, lon: float | None = None,
                use_rerank: bool = True) -> Prediction:
        seed = sum(len(p) for p in paths) + len(paths)
        rng = np.random.default_rng(seed)
        logits = rng.normal(size=len(self.taxa)) * (1.5 + len(paths))
        p = np.exp(logits - logits.max()); p /= p.sum()
        order = np.argsort(-p)
        pred_set = [int(i) for i in order if p[i] >= 0.1] or [int(order[0])]
        top = self.taxa[int(order[0])]
        if p[order[0]] >= 0.6:
            rank, label, conf = "species", top.species, float(p[order[0]])
        else:
            rank, label = "genus", top.genus
            conf = float(sum(p[t.idx] for t in self.taxa if t.genus == top.genus))
        return Prediction(
            probs=p, top_k=[(int(i), float(p[i])) for i in order[:5]], pred_set=pred_set,
            rank=rank, label=label, confidence=conf,
            evidence=[f"mock/{top.species.replace(' ', '_')}/{k}.jpg" for k in range(3)],
            traits=["ovate leaf with long drip tip", "entire margin", "prominent pale midrib"],
            organ="leaf", reranked=len(pred_set) > 1 and use_rerank,
            rationale="mock: drip tip and midrib match the top candidate" if len(pred_set) > 1 else "",
        )

    def add_species(self, taxon: Taxon, paths: list[str]) -> None:
        self.taxa.append(Taxon(len(self.taxa), taxon.species, taxon.genus, taxon.family))


def get_pipeline(dataset: str = "inat_india"):
    """VANALENS_MOCK=1 -> MockPipeline, else the real VanaLens."""
    if os.environ.get("VANALENS_MOCK") == "1":
        return MockPipeline()
    return VanaLens.from_cache(dataset)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VanaLens integration: fit geo beta, run end-to-end photos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fb = sub.add_parser("fit-beta")
    fb.add_argument("--dataset", required=True)
    e2e = sub.add_parser("e2e")
    e2e.add_argument("--dataset", required=True)
    e2e.add_argument("--n", type=int, default=20)
    e2e.add_argument("--manifest", type=Path, default=None)
    e2e.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "fit-beta":
        print(json.dumps(fit_beta_cached(a.dataset)))
    else:
        out = run_e2e(a.dataset, a.n, a.manifest, use_rerank=not a.no_rerank)
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
