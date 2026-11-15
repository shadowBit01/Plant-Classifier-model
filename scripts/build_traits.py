"""Lane L4 (context). Generate plant visual traits and save text prototypes.

Usage:
  python scripts/build_traits.py [--dataset inat_india] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from vanalens import config, contracts
from vanalens.traits import build_text_protos, generate_traits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_or_create_taxa(dataset: str, labels_path: Path, data_dir: Path) -> list[contracts.Taxon]:
    """Load taxa from labels.json, or discover from dataset subdirectories if labels.json is not yet built."""
    if labels_path.exists():
        logger.info("Loading taxa from %s", labels_path)
        return contracts.load_taxa(labels_path)

    dataset_dir = data_dir / dataset
    if dataset_dir.exists():
        species_dirs = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
        if species_dirs:
            logger.info("Found %d species folders under %s; building taxa", len(species_dirs), dataset_dir)
            taxa = [
                contracts.Taxon(
                    idx=i,
                    species=sp,
                    genus=sp.split()[0] if " " in sp else sp,
                    family="",
                )
                for i, sp in enumerate(species_dirs)
            ]
            contracts.save_taxa(taxa, labels_path)
            return taxa

    # Fallback to mock taxa if dataset does not exist yet
    from vanalens.pipeline import MockPipeline
    logger.warning("Dataset not found at %s; using mock taxa", dataset_dir)
    return list(MockPipeline.TAXA)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate traits and text prototypes for a dataset.")
    parser.add_argument("--dataset", type=str, default="inat_india", help="Dataset name")
    parser.add_argument("--labels", type=Path, default=None, help="Optional path to labels.json")
    parser.add_argument("--out", type=Path, default=None, help="Output .npz path for text prototypes")
    parser.add_argument("--traits-cache", type=Path, default=None, help="Optional traits cache path")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (default: cpu)")
    parser.add_argument("--organ", type=str, default="leaf or flower", help="Organ to focus on")
    parser.add_argument("--n", type=int, default=6, help="Number of traits per species (default: 6)")
    args = parser.parse_args(argv)

    labels_path = args.labels or (config.DATA_DIR / args.dataset / "labels.json")
    out_path = args.out or (config.CACHE_DIR / "text" / f"{args.dataset}.npz")
    traits_cache = args.traits_cache or (config.CACHE_DIR / "traits.json")

    taxa = find_or_create_taxa(args.dataset, labels_path, config.DATA_DIR)
    logger.info("Loaded %d taxa for dataset '%s'", len(taxa), args.dataset)

    traits = generate_traits(taxa, organ=args.organ, n=args.n, cache_path=traits_cache)
    logger.info("Retrieved traits for %d species (cached at %s)", len(traits), traits_cache)

    # Print 2 example species only as required by card L4-b
    print("\n" + "=" * 50)
    print("Example species traits (showing 2 species):")
    print("=" * 50)
    example_taxa = taxa[:2]
    for t in example_taxa:
        sp_traits = traits.get(t.species, [])
        print(f"\nSpecies: {t.species} (Genus: {t.genus})")
        for idx, trait in enumerate(sp_traits, start=1):
            print(f"  {idx}. {trait}")
    print("=" * 50 + "\n")

    # Encode text prototypes using BioCLIP 2 text tower via vanalens.encoders.Encoder
    logger.info("Loading Encoder for text prototype encoding on %s...", args.device)
    from vanalens.encoders import Encoder
    encoder = Encoder(device=args.device, fp16=False)

    logger.info("Building text prototypes for %d classes...", len(taxa))
    protos, prompt_strings = build_text_protos(taxa, traits, encoder.encode_text)

    # Save to $VANALENS_CACHE/text/<dataset>.npz
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.array([len(p) for p in protos], dtype=np.int32)
    uniform = len(set(counts)) == 1

    np.savez(
        out_path,
        protos=np.array(protos, dtype=object),
        strings=np.array(prompt_strings, dtype=object),
        stacked_protos=np.stack(protos) if uniform else np.vstack(protos),
        counts=counts,
        species=np.array([t.species for t in taxa]),
        strings_json=json.dumps(prompt_strings),
        traits_json=json.dumps({t.species: traits.get(t.species, []) for t in taxa}),
    )
    logger.info("Saved text prototypes to %s (size: %s bytes)", out_path, out_path.stat().st_size)


if __name__ == "__main__":
    main()
