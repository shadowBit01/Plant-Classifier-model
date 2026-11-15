"""Lane L4 (context). Baseline B4: open-ended VLM zero-shot identification.

Usage:
  python scripts/baseline_vlm.py [--dataset inat_india] [--n 100] [--seed 0]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from vanalens import config, contracts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _normalize_name(text: str) -> str:
    """Extract binomial and normalize: lowercase, strip punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_latin_binomial(text: str) -> tuple[str, str]:
    """Attempt to extract (species, genus) from an open-ended VLM reply."""
    # Find words looking like Capitalized lowercase (Genus species)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = _normalize_name(text)

    # Check common patterns like 'Latin binomial: Genus species' or first two words
    words = cleaned.split()
    if len(words) >= 2:
        genus = words[0]
        species = f"{words[0]} {words[1]}"
        return species, genus
    elif len(words) == 1:
        return words[0], words[0]
    return "", ""


def _encode_image_b64(path: str) -> tuple[str, str]:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else "jpeg"
    media_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }
    return data, media_types.get(ext, "image/jpeg")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Baseline B4: open-ended VLM identification.")
    parser.add_argument("--dataset", type=str, default="inat_india", help="Dataset name")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional test manifest path")
    parser.add_argument("--labels", type=Path, default=None, help="Optional labels.json path")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Raw response cache directory")
    parser.add_argument("--n", type=int, default=100, help="Number of test images (default: 100)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    args = parser.parse_args(argv)

    out_path = args.out or (config.RUNS_DIR / "L4" / f"b4_{args.dataset}.json")
    cache_dir = args.cache_dir or (config.CACHE_DIR / "vlm")
    manifest_path = args.manifest or (config.CACHE_DIR / "manifests" / f"{args.dataset}_test.jsonl")
    labels_path = args.labels or (config.DATA_DIR / args.dataset / "labels.json")

    api_key = os.environ.get(config.LLM_KEY_ENV)
    if not api_key:
        logger.warning("%s is not set. Skipping baseline B4 as instructed.", config.LLM_KEY_ENV)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "n": 0,
                    "species_accuracy": None,
                    "genus_accuracy": None,
                    "status": "skipped",
                    "reason": f"{config.LLM_KEY_ENV} is empty: reranker and B4 disabled",
                },
                indent=2,
            )
        )
        print("reranker and B4 disabled: no key")
        return

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    model = os.environ.get("VANALENS_LLM_MODEL") or "claude-3-5-sonnet-20241022"

    if not manifest_path.exists():
        logger.error("Test manifest not found at %s", manifest_path)
        sys.exit(1)

    samples: list[contracts.Sample] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(contracts.Sample(**json.loads(line)))

    if not samples:
        logger.error("No samples found in %s", manifest_path)
        sys.exit(1)

    taxa: list[contracts.Taxon] = contracts.load_taxa(labels_path)
    label_to_taxon = {t.idx: t for t in taxa}

    # Sample 100 test images with seed 0
    rng = np.random.default_rng(args.seed)
    n_sample = min(args.n, len(samples))
    chosen_indices = rng.choice(len(samples), size=n_sample, replace=False)
    chosen_samples = [samples[i] for i in chosen_indices]

    cache_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = "What plant species is this? Answer with the Latin binomial."

    species_correct = 0
    genus_correct = 0
    total = 0

    for sample in chosen_samples:
        if sample.label is None or sample.label not in label_to_taxon:
            continue
        gt = label_to_taxon[sample.label]
        img_hash = hashlib.sha256(sample.path.encode("utf-8")).hexdigest()[:16]
        cache_file = cache_dir / f"{img_hash}.json"

        raw_answer: str = ""
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                raw_answer = cached.get("answer", "")
            except Exception:
                raw_answer = ""

        if not raw_answer:
            try:
                b64_data, media_type = _encode_image_b64(sample.path)
                msg = client.messages.create(
                    model=model,
                    max_tokens=128,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": media_type, "data": b64_data},
                                },
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ],
                )
                raw_answer = msg.content[0].text
                cache_file.write_text(
                    json.dumps({"path": sample.path, "answer": raw_answer}, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.warning("Error querying VLM for %s: %s", sample.path, exc)
                raw_answer = ""

        pred_species, pred_genus = _extract_latin_binomial(raw_answer)
        gt_species = _normalize_name(gt.species)
        gt_genus = _normalize_name(gt.genus)

        if pred_species == gt_species:
            species_correct += 1
        if pred_genus == gt_genus:
            genus_correct += 1
        total += 1

    species_acc = (species_correct / total) if total > 0 else 0.0
    genus_acc = (genus_correct / total) if total > 0 else 0.0

    result = {
        "dataset": args.dataset,
        "n": total,
        "seed": args.seed,
        "species_accuracy": round(species_acc, 4),
        "genus_accuracy": round(genus_acc, 4),
        "model": model,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved B4 results to %s: %s", out_path, result)


if __name__ == "__main__":
    main()
