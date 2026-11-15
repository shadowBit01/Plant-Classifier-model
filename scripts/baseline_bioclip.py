"""Baseline B2 (Lane L2): BioCLIP 2 zero-shot evaluation via pybioclip.

Runs:
1. Closed-set pybioclip CustomLabelsClassifier over the dataset's species names on a test manifest
   -> (N, C) probability matrix in taxa order -> eval.harness -> runs/L2/b2_<dataset>.json.
2. Open-ended pybioclip TreeOfLifeClassifier over the same test manifest, recording top-1 and top-5
   species-name match rates in runs/L2/b2_<dataset>.json.
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import bioclip.predict as bp
import numpy as np
import torch
from bioclip import CustomLabelsClassifier, Rank, TreeOfLifeClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import harness
from vanalens import config, contracts, data
from vanalens.encoders import _filter_readable_samples, _read_manifest_samples

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple[str, str | None, str], tuple[torch.nn.Module, object]] = {}
_TOL_TXT_CACHE: dict[tuple[str, str], torch.Tensor] = {}
_PATH_FEAT_CACHE: dict[str, np.ndarray] = {}
_ORIG_LOAD_PRETRAINED = bp.BaseClassifier.load_pretrained_model


def _prepopulate_path_feat_cache(dataset: str, samples: list[contracts.Sample]) -> None:
    """If $VANALENS_CACHE/emb/<dataset>_test.npz exists, pre-populate _PATH_FEAT_CACHE from its BioCLIP block."""
    emb_path = config.CACHE_DIR / "emb" / f"{dataset}_test.npz"
    if not emb_path.exists():
        return
    try:
        with np.load(emb_path, allow_pickle=False) as d:
            ids = [str(x) for x in d["ids"]]
            X_src = d["X_plain"] if "X_plain" in d and d["X_plain"].shape == d["X"].shape else d["X"]
            bio_block = np.asarray(X_src[:, : config.BIOCLIP_DIM], dtype=np.float32)
        id_to_path = {s.image_id: s.path for s in samples}
        for idx, img_id in enumerate(ids):
            p = id_to_path.get(img_id)
            if p is not None:
                _PATH_FEAT_CACHE[p] = bio_block[idx]
    except Exception as exc:
        logger.warning("Could not pre-populate BioCLIP feature cache from %s: %s", emb_path, exc)


def _get_or_encode_bio_features(clf: bp.BaseClassifier, batch_paths: list[str]) -> torch.Tensor:
    """Return (B, BIOCLIP_DIM) L2-normalised BioCLIP image features on clf.device, using _PATH_FEAT_CACHE when available."""
    missing_indices = [i for i, p in enumerate(batch_paths) if p not in _PATH_FEAT_CACHE]
    if missing_indices:
        missing_paths = [batch_paths[i] for i in missing_indices]
        pil_imgs = [clf.ensure_rgb_image(p) for p in missing_paths]
        use_fp16 = str(clf.device).startswith("cuda")
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            feats = clf.create_image_features(pil_imgs).float().cpu().numpy()
        for idx, p in zip(missing_indices, missing_paths):
            _PATH_FEAT_CACHE[p] = feats[missing_indices.index(idx)] if False else None
        for feat_row, p in zip(feats, missing_paths):
            _PATH_FEAT_CACHE[p] = feat_row.astype(np.float32)

    arr = np.stack([_PATH_FEAT_CACHE[p] for p in batch_paths], axis=0)
    use_fp16 = str(clf.device).startswith("cuda")
    dtype = torch.float16 if use_fp16 else torch.float32
    return torch.from_numpy(arr).to(device=clf.device, dtype=dtype)


def _install_bioclip_memory_patches() -> None:
    """Patch pybioclip to reuse the frozen BioCLIP 2 backbone and stream TreeOfLife embeddings in fp16 on CUDA."""

    def _cached_load_pretrained_model(
        self: bp.BaseClassifier,
        model_str: str = bp.BIOCLIP_MODEL_STR,
        pretrained_str: str | None = None,
    ) -> None:
        resolved_model = model_str or bp.BIOCLIP_MODEL_STR
        key = (resolved_model, pretrained_str, str(self.device))
        if key in _MODEL_CACHE:
            self.model_str = resolved_model
            self.pretrained_str = pretrained_str
            self.model, self.preprocess = _MODEL_CACHE[key]
            return
        _ORIG_LOAD_PRETRAINED(self, model_str=model_str, pretrained_str=pretrained_str)
        self.model.eval()
        self.model.requires_grad_(False)
        _MODEL_CACHE[key] = (self.model, self.preprocess)

    def _streamed_get_txt_emb(self: TreeOfLifeClassifier) -> torch.Tensor:
        txt_emb_npy = self.get_cached_datafile("embeddings/txt_emb_species.npy")
        cache_key = (str(txt_emb_npy), str(self.device))
        if cache_key in _TOL_TXT_CACHE:
            return _TOL_TXT_CACHE[cache_key]
        use_fp16 = str(self.device).startswith("cuda")
        dtype = torch.float16 if use_fp16 else torch.float32
        with open(txt_emb_npy, "rb") as f:
            ver = np.lib.format.read_magic(f)
            if ver == (1, 0):
                shape, fortran_order, _ = np.lib.format.read_array_header_1_0(f)
            else:
                shape, fortran_order, _ = np.lib.format.read_array_header_2_0(f)
            if fortran_order:
                arr = np.load(txt_emb_npy)
                out = torch.from_numpy(arr).to(device=self.device, dtype=dtype)
                _TOL_TXT_CACHE[cache_key] = out
                return out
            out = torch.empty(shape, dtype=dtype, device=self.device)
            row_chunk = 16384
            for start_row in range(0, shape[0], row_chunk):
                n_rows = min(row_chunk, shape[0] - start_row)
                raw = bytearray(f.read(n_rows * shape[1] * 4))
                chunk_tensor = torch.frombuffer(raw, dtype=torch.float32).reshape(n_rows, shape[1])
                out[start_row : start_row + n_rows].copy_(chunk_tensor.to(device=self.device, dtype=dtype))
        _TOL_TXT_CACHE[cache_key] = out
        return out

    def _mixed_dtype_create_probabilities(
        self: bp.BaseClassifier,
        img_features: torch.Tensor,
        txt_features: torch.Tensor,
    ) -> torch.Tensor:
        if img_features.dtype != txt_features.dtype:
            img_features = img_features.to(dtype=txt_features.dtype)
        scale = self.model.logit_scale.exp().to(dtype=txt_features.dtype)
        logits = (scale * (img_features @ txt_features)).float()
        return torch.nn.functional.softmax(logits, dim=1)

    orig_get_txt_embeddings = bp.CustomLabelsClassifier._get_txt_embeddings

    @torch.no_grad()
    def _fast_get_txt_embeddings(self: bp.CustomLabelsClassifier, classnames: list[str]) -> torch.Tensor:
        use_fp16 = str(self.device).startswith("cuda")
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            return orig_get_txt_embeddings(self, classnames)

    bp.BaseClassifier.load_pretrained_model = _cached_load_pretrained_model
    bp.TreeOfLifeClassifier.get_txt_emb = _streamed_get_txt_emb
    bp.BaseClassifier.create_probabilities = _mixed_dtype_create_probabilities
    bp.CustomLabelsClassifier._get_txt_embeddings = _fast_get_txt_embeddings


def predict_custom_labels_proba(
    paths: list[str],
    taxa: list[contracts.Taxon],
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    """Run pybioclip CustomLabelsClassifier over species names and return (N, C) probabilities in taxa order."""
    n_classes = len(taxa)
    if not paths:
        return np.zeros((0, n_classes), dtype=np.float64)

    species_names = [t.species.strip() for t in taxa]
    species_to_idx = {name: t.idx for name, t in zip(species_names, taxa)}

    clf = CustomLabelsClassifier(cls_ary=species_names, device=device)
    P = np.zeros((len(paths), n_classes), dtype=np.float64)

    if isinstance(clf, bp.CustomLabelsClassifier):
        col_order = [species_to_idx[c.strip()] for c in clf.classes]
        with torch.inference_mode():
            txt_features = clf.txt_embeddings
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                img_features = _get_or_encode_bio_features(clf, batch_paths)
                probs = clf.create_probabilities(img_features, txt_features).detach().cpu().numpy()
                P[start : start + len(batch_paths), col_order] = probs
    else:
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            preds = clf.predict(batch_paths, k=n_classes, batch_size=len(batch_paths))
            for i in range(len(batch_paths)):
                row_preds = preds[i * n_classes : (i + 1) * n_classes]
                for item in row_preds:
                    cls_name = item["classification"].strip()
                    cls_idx = species_to_idx[cls_name]
                    P[start + i, cls_idx] = float(item["score"])

    row_sums = P.sum(axis=1, keepdims=True)
    nonzero = row_sums.squeeze(-1) > 0
    if np.any(nonzero):
        P[nonzero] /= row_sums[nonzero]
    if np.any(~nonzero):
        P[~nonzero] = 1.0 / n_classes

    contracts.check_proba(P, n_classes)
    return P


def evaluate_tree_of_life_open_ended(
    paths: list[str],
    y: np.ndarray,
    taxa: list[contracts.Taxon],
    device: str = "cuda",
    batch_size: int = 64,
) -> dict[str, float]:
    """Run pybioclip TreeOfLifeClassifier at Rank.SPECIES and return top-1 and top-5 species-name match rates."""
    if not paths:
        return {"tol_top1": 0.0, "tol_top5": 0}

    tol = TreeOfLifeClassifier(device=device)
    k = 5
    top1_hits = 0
    top5_hits = 0

    if isinstance(tol, bp.TreeOfLifeClassifier):
        with torch.inference_mode():
            txt_features = tol.get_txt_embeddings()
            k_eff = min(k, txt_features.shape[1])
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                batch_y = y[start : start + batch_size]
                img_features = _get_or_encode_bio_features(tol, batch_paths)
                if img_features.dtype != txt_features.dtype:
                    img_features = img_features.to(dtype=txt_features.dtype)
                scores = img_features @ txt_features
                topk_indices = scores.topk(k_eff, dim=1).indices.cpu().tolist()
                for row_indices, true_idx in zip(topk_indices, batch_y):
                    target_species = taxa[int(true_idx)].species.strip().lower()
                    pred_species = [
                        str(tol.get_classification_dict(idx, Rank.SPECIES).get("species", "")).strip().lower()
                        for idx in row_indices
                    ]
                    if pred_species and pred_species[0] == target_species:
                        top1_hits += 1
                    if target_species in pred_species[:5]:
                        top5_hits += 1
    else:
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch_y = y[start : start + batch_size]
            preds = tol.predict(batch_paths, rank=Rank.SPECIES, k=k, batch_size=len(batch_paths))
            for i, true_idx in enumerate(batch_y):
                target_species = taxa[int(true_idx)].species.strip().lower()
                row_preds = preds[i * k : (i + 1) * k]
                pred_species = [str(item.get("species", "")).strip().lower() for item in row_preds]
                if pred_species and pred_species[0] == target_species:
                    top1_hits += 1
                if target_species in pred_species[:5]:
                    top5_hits += 1

    n = len(paths)
    return {
        "tol_top1": round(top1_hits / n, 4),
        "tol_top5": round(top5_hits / n, 4),
    }


def run_dataset(
    dataset: str,
    manifest_path: Path | None = None,
    labels_path: Path | None = None,
    device: str = "cuda",
    batch_size: int = 64,
    run_tol: bool = True,
) -> dict:
    """Evaluate B2 (CustomLabelsClassifier + TreeOfLifeClassifier) on a dataset and write runs/L2/b2_<dataset>.json."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    _install_bioclip_memory_patches()

    manifest = manifest_path or (config.CACHE_DIR / "manifests" / f"{dataset}_test.jsonl")
    labels = labels_path or (config.DATA_DIR / dataset / "labels.json")

    taxa = contracts.load_taxa(labels)
    raw_samples = _read_manifest_samples(manifest)
    valid_samples = [s for s in _filter_readable_samples(raw_samples) if s.label is not None]
    _prepopulate_path_feat_cache(dataset, valid_samples)

    paths = [s.path for s in valid_samples]
    y = np.array([int(s.label) for s in valid_samples], dtype=np.int64)

    t0 = time.perf_counter()
    P = predict_custom_labels_proba(paths, taxa, device=device, batch_size=batch_size)
    metrics = harness.evaluate(P, y, taxa)

    if run_tol:
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        tol_metrics = evaluate_tree_of_life_open_ended(paths, y, taxa, device=device, batch_size=batch_size)
        metrics["tol_top1"] = tol_metrics["tol_top1"]
        metrics["tol_top5"] = tol_metrics["tol_top5"]
        metrics["open_ended_top1"] = tol_metrics["tol_top1"]
        metrics["open_ended_top5"] = tol_metrics["tol_top5"]
        metrics["tree_of_life"] = {
            "top1": tol_metrics["tol_top1"],
            "top5": tol_metrics["tol_top5"],
        }

    elapsed = max(time.perf_counter() - t0, 1e-6)
    metrics["dataset"] = dataset
    metrics["baseline"] = "B2"
    metrics["elapsed_sec"] = round(elapsed, 2)

    out_path = harness.save_result(metrics, f"b2_{dataset}", "L2")
    logger.info(
        "B2 %s (n=%d): custom top1=%.4f top5=%.4f macro_top1=%.4f | TOL top1=%.4f top5=%.4f -> %s",
        dataset,
        metrics["n"],
        metrics["top1"],
        metrics["top5"],
        metrics["macro_top1"],
        metrics.get("tol_top1", 0.0),
        metrics.get("tol_top5", 0.0),
        out_path,
    )
    return metrics


def _discover_datasets() -> list[str]:
    manifests_dir = config.CACHE_DIR / "manifests"
    if not manifests_dir.exists():
        return []
    datasets: list[str] = []
    for p in sorted(manifests_dir.glob("*_test.jsonl")):
        ds = p.name[: -len("_test.jsonl")]
        if (config.DATA_DIR / ds / "labels.json").exists():
            datasets.append(ds)
    return datasets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run BioCLIP 2 zero-shot baseline (B2).")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (default: all with test manifests)")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional explicit test manifest path")
    parser.add_argument("--labels", type=Path, default=None, help="Optional explicit labels.json path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--skip-tol", action="store_true", help="Skip open-ended TreeOfLifeClassifier evaluation")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = _discover_datasets()
        if not datasets:
            raise FileNotFoundError(f"No *_test.jsonl manifests found in {config.CACHE_DIR / 'manifests'}")

    for ds in datasets:
        res = run_dataset(
            dataset=ds,
            manifest_path=args.manifest,
            labels_path=args.labels,
            device=args.device,
            batch_size=args.batch_size,
            run_tol=not args.skip_tol,
        )
        print(
            f"{ds}: n={res['n']} top1={res['top1']:.4f} top5={res['top5']:.4f} "
            f"macro_top1={res['macro_top1']:.4f} tol_top1={res.get('tol_top1', 0.0):.4f} "
            f"tol_top5={res.get('tol_top5', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
