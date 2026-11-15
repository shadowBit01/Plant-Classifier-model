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

    # Load common names from species.csv if present
    common_names: dict[str, str] = {}
    species_csv = config.DATA_DIR / args.dataset / "species.csv"
    if species_csv.exists():
        try:
            import pandas as pd
            df_sp = pd.read_csv(species_csv)
            if "name" in df_sp.columns and "common_name" in df_sp.columns:
                for _, row in df_sp.iterrows():
                    val = row["common_name"]
                    if pd.notna(val) and str(val).strip():
                        common_names[str(row["name"]).strip()] = str(val).strip()
            logger.info("Loaded %d common names from %s", len(common_names), species_csv)
        except Exception as exc:
            logger.warning("Could not parse species.csv: %s", exc)

    traits = generate_traits(
        taxa,
        organ=args.organ,
        n=args.n,
        cache_path=traits_cache,
        common_names=common_names,
    )
    logger.info("Retrieved traits for %d species (cached at %s)", len(traits), traits_cache)

    species_with_traits = [t for t in taxa if t.species in traits and traits[t.species]]
    fallback_taxa = [t for t in taxa if t.species not in traits or not traits[t.species]]
    logger.info(
        "Species with traits: %d, taxonomic-fallback count: %d (total: %d)",
        len(species_with_traits), len(fallback_taxa), len(taxa)
    )

    # Print 2 example species only as required by card L4-b
    print("\n" + "=" * 50)
    print("Example species traits (showing 2 species):")
    print("=" * 50)
    example_taxa = [t for t in taxa if t.species in traits][:2]
    for t in example_taxa:
        sp_traits = traits.get(t.species, [])
        print(f"\nSpecies: {t.species} (Genus: {t.genus})")
        for idx, trait in enumerate(sp_traits, start=1):
            print(f"  {idx}. {trait}")
    print("=" * 50 + "\n")

    if fallback_taxa:
        from vanalens.traits import taxonomic_prompt
        print("=" * 50)
        print(f"Taxonomic fallback species ({len(fallback_taxa)} species):")
        print("=" * 50)
        for t in fallback_taxa:
            cname = common_names.get(t.species)
            prompt_str = taxonomic_prompt(t, cname)
            print(f"  [{t.idx}] {t.species} -> \"{prompt_str}\"")
        print("=" * 50 + "\n")

    # Encode text prototypes using BioCLIP 2 text tower via vanalens.encoders.Encoder
    logger.info("Loading Encoder for text prototype encoding on %s...", args.device)
    from vanalens.encoders import Encoder
    encoder = Encoder(device=args.device, fp16=False)

    logger.info("Building text prototypes for %d classes...", len(taxa))
    protos, prompt_strings = build_text_protos(
        taxa,
        traits,
        encoder.encode_text,
        common_names=common_names,
        taxonomic_fallback=True,
    )
    assert len(protos) == len(taxa), f"Expected {len(taxa)} prototypes, got {len(protos)}"

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

