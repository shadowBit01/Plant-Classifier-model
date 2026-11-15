"""Baseline B0: Pl@ntNet API v2 identify evaluation on a test manifest.

Samples up to 400 images (seed 0) from a test manifest, calls the Pl@ntNet API v2
identify endpoint (key from env PLANTNET_API_KEY, passing organ when known), caches
each raw JSON under $VANALENS_CACHE/plantnet_api/, maps returned species names to
taxa idx (unmatched -> uniform row), and saves results with eval.harness to
runs/L1/b0_<dataset>.json. Never exceeds 450 calls per day and skips cached images.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import mimetypes
import os
from pathlib import Path
import re
import sys

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import evaluate, save_result
from vanalens.config import CACHE_DIR, DATA_DIR
from vanalens.contracts import Sample, Taxon, check_proba, load_taxa
from vanalens.data import read_manifest

PLANTNET_API_URL = "https://my-api.plantnet.org/v2/identify/{project}"
PLANTNET_ORGANS = {"leaf", "flower", "fruit", "bark"}
MAX_DAILY_CALLS = 450
DEFAULT_SAMPLE_SIZE = 400


def sample_manifest(
    samples: list[Sample],
    max_samples: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 0,
) -> list[Sample]:
    """Deterministically sample up to `max_samples` labeled images (seed 0)."""
    labeled = [s for s in samples if s.label is not None]
    if len(labeled) <= max_samples:
        return list(labeled)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(labeled), size=max_samples, replace=False))
    return [labeled[int(i)] for i in chosen]


def plantnet_organ(organ: str | None) -> str:
    """Map VanaLens organ tag to Pl@ntNet API organ parameter ('leaf'|'flower'|'fruit'|'bark'|'auto')."""
    if organ is not None:
        norm = organ.strip().lower()
        if norm in PLANTNET_ORGANS:
            return norm
    return "auto"


def cache_path_for(sample: Sample, cache_dir: Path) -> Path:
    """Return deterministic cache path under `cache_dir` for a sample."""
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", sample.image_id).strip("_")
    if not safe_id:
        safe_id = Path(sample.path).stem
    return Path(cache_dir) / f"{safe_id}.json"


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_daily_count(cache_dir: Path, today: str | None = None) -> int:
    today = today or _today_utc()
    ledger = Path(cache_dir) / "_daily_calls.json"
    if not ledger.exists():
        return 0
    try:
        data = json.loads(ledger.read_text(encoding="utf-8"))
        if data.get("date") == today:
            return int(data.get("count", 0))
    except Exception:
        pass
    return 0


def _save_daily_count(cache_dir: Path, count: int, today: str | None = None) -> None:
    today = today or _today_utc()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    ledger = Path(cache_dir) / "_daily_calls.json"
    ledger.write_text(
        json.dumps({"date": today, "count": int(count)}, indent=2),
        encoding="utf-8",
    )


def _canonical_binomial(name: str) -> str:
    parts = name.strip().lower().split()
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return " ".join(parts)


def response_to_proba(resp: dict | None, taxa: list[Taxon]) -> np.ndarray:
    """Map a Pl@ntNet API response dict to a (C,) probability vector (unmatched -> uniform)."""
    n_classes = len(taxa)
    if n_classes == 0:
        return np.zeros(0, dtype=float)
    uniform = np.full(n_classes, 1.0 / n_classes, dtype=float)
    if not resp or not isinstance(resp, dict):
        return uniform

    results = resp.get("results")
    if not isinstance(results, list) or not results:
        return uniform

    sp_to_idx: dict[str, int] = {}
    gbif_to_idx: dict[str, int] = {}
    for t in taxa:
        sp_to_idx[t.species.strip().lower()] = t.idx
        sp_to_idx[_canonical_binomial(t.species)] = t.idx
        if t.gbif_key is not None:
            gbif_to_idx[str(t.gbif_key)] = t.idx

    row = np.zeros(n_classes, dtype=float)
    for item in results:
        if not isinstance(item, dict):
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if score <= 0.0 or not np.isfinite(score):
            continue

        species_info = item.get("species") or {}
        idx: int | None = None
        if isinstance(species_info, dict):
            raw_name = str(
                species_info.get("scientificNameWithoutAuthor")
                or species_info.get("scientificName")
                or ""
            ).strip()
            if raw_name:
                idx = sp_to_idx.get(raw_name.lower())
                if idx is None:
                    idx = sp_to_idx.get(_canonical_binomial(raw_name))

        if idx is None:
            gbif_info = item.get("gbif")
            if isinstance(gbif_info, dict) and gbif_info.get("id") is not None:
                idx = gbif_to_idx.get(str(gbif_info["id"]))

        if idx is not None:
            row[idx] += score

    total = float(row.sum())
    if total <= 0.0 or not np.isfinite(total):
        return uniform
    return row / total


def call_plantnet_identify(
    image_path: str | Path,
    api_key: str,
    organ: str | None = None,
    project: str = "all",
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> dict:
    """Call Pl@ntNet v2 identify endpoint for a single image and return parsed JSON."""
    img_p = Path(image_path)
    mime_type = mimetypes.guess_type(img_p.name)[0] or "image/jpeg"
    organ_tag = plantnet_organ(organ)
    url = PLANTNET_API_URL.format(project=project)
    params = {"api-key": api_key}
    data = {"organs": [organ_tag]}

    http = session or requests
    with img_p.open("rb") as f:
        files = [("images", (img_p.name, f, mime_type))]
        resp = http.post(url, params=params, files=files, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def run_b0(
    dataset: str = "inat_india",
    *,
    manifest_path: Path | None = None,
    labels_path: Path | None = None,
    cache_dir: Path | None = None,
    api_key: str | None = None,
    project: str = "all",
    max_samples: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 0,
    daily_limit: int = MAX_DAILY_CALLS,
    session: requests.Session | None = None,
) -> dict | None:
    """Run Baseline B0 on `dataset`, caching responses and writing runs/L1/b0_<dataset>.json.

    Returns the evaluation result dict, or None if skipped due to missing API key.
    """
    key = (api_key if api_key is not None else os.environ.get("PLANTNET_API_KEY", "")).strip()
    if not key:
        print("B0 skipped: no key")
        return None

    manifest_file = Path(manifest_path or (CACHE_DIR / "manifests" / f"{dataset}_test.jsonl"))
    labels_file = Path(labels_path or (DATA_DIR / dataset / "labels.json"))
    out_cache_dir = Path(cache_dir or (CACHE_DIR / "plantnet_api"))
    out_cache_dir.mkdir(parents=True, exist_ok=True)

    taxa = load_taxa(labels_file)
    all_samples = read_manifest(manifest_file)
    sampled = sample_manifest(all_samples, max_samples=max_samples, seed=seed)

    effective_limit = min(int(daily_limit), MAX_DAILY_CALLS)
    today = _today_utc()
    calls_today = _load_daily_count(out_cache_dir, today=today)

    rows: list[np.ndarray] = []
    y_list: list[int] = []

    for s in sampled:
        if s.label is None:
            continue
        cpath = cache_path_for(s, out_cache_dir)
        raw_resp: dict | None = None
        if cpath.exists():
            try:
                cached = json.loads(cpath.read_text(encoding="utf-8"))
                if isinstance(cached, dict):
                    raw_resp = cached
            except Exception:
                raw_resp = None

        if raw_resp is None and calls_today < effective_limit:
            try:
                raw_resp = call_plantnet_identify(
                    s.path,
                    api_key=key,
                    organ=s.organ,
                    project=project,
                    session=session,
                )
                cpath.write_text(json.dumps(raw_resp, indent=2), encoding="utf-8")
            except Exception as exc:
                raw_resp = {"error": str(exc), "results": []}
            calls_today += 1
            _save_daily_count(out_cache_dir, calls_today, today=today)

        rows.append(response_to_proba(raw_resp, taxa))
        y_list.append(int(s.label))

    P = np.vstack(rows) if rows else np.zeros((0, len(taxa)), dtype=float)
    y = np.asarray(y_list, dtype=np.int64)
    if len(P) > 0:
        check_proba(P, len(taxa))

    res = evaluate(P, y, taxa)
    save_result(res, f"b0_{dataset}", "L1")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline B0: Pl@ntNet API v2 identify")
    parser.add_argument("--dataset", default="inat_india", help="Dataset name (default: inat_india)")
    parser.add_argument("--manifest", default=None, help="Override path to test manifest JSONL")
    parser.add_argument("--labels", default=None, help="Override path to labels.json")
    parser.add_argument("--project", default="all", help="Pl@ntNet flora project (default: all)")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_SAMPLE_SIZE, help="Max test images to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling (default: 0)")
    args = parser.parse_args()

    res = run_b0(
        dataset=args.dataset,
        manifest_path=Path(args.manifest) if args.manifest else None,
        labels_path=Path(args.labels) if args.labels else None,
        project=args.project,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if res is not None:
        print(
            f"b0_{args.dataset}: n={res['n']} top1={res['top1']} top5={res['top5']} macro_top1={res['macro_top1']}"
        )


if __name__ == "__main__":
    main()
