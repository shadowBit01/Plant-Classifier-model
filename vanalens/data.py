"""Lane L1 (data-eval). Dataset loaders, grouped splits, manifests. Tests: tests/test_eval.py

Expected raw layouts under DATA_DIR (inspect with `find $VANALENS_DATA/<name> -maxdepth 2 | head`):
  dimpsar/        as downloaded from Mendeley (10.17632/748f8jkphb), folder per species
  plantnet300k/   images/{train,val,test}/<species_id>/*.jpg + plantnet300K_species_id_2_name.json
  campus/         <Genus species>/*.jpg, optional meta.csv (image,lat,lon,organ,plant_id)
Every loader returns (samples, taxa) with taxa idx 0..C-1 sorted by species name, and writes
DATA_DIR/<name>/labels.json via contracts.save_taxa. Genus = first word of species; family from a
lookup (GBIF species/match, cached) or "" if unknown.
"""
from __future__ import annotations

from pathlib import Path

from .contracts import Sample, Taxon


def load_dimpsar(root: Path, part: str = "leaf") -> tuple[list[Sample], list[Taxon]]:
    """part: "leaf" (80 species single leaves) or "plant" (40 species). group = image stem
    unless the dataset exposes plant/session ids."""
    raise NotImplementedError


def load_plantnet300k(root: Path, split: str) -> tuple[list[Sample], list[Taxon]]:
    """split in {"train", "val", "test"}; label space = all 1,081 species, same idx for every split."""
    raise NotImplementedError


def load_campus(root: Path, taxa: list[Taxon] | None = None) -> tuple[list[Sample], list[Taxon]]:
    """If `taxa` is given, map folder names onto it (unknown species -> label None). group = plant_id or folder.
    Fill lat/lon from meta.csv, else from EXIF (read_exif_gps)."""
    raise NotImplementedError


def grouped_split(samples: list[Sample], fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
                  seed: int = 0) -> dict[str, list[Sample]]:
    """Keys "gallery", "cal", "test". No group may appear in two splits. Stratify by label where possible
    so every class with >= 3 groups appears in all splits."""
    raise NotImplementedError


def cap_per_class(samples: list[Sample], cap: int = 100, seed: int = 0) -> list[Sample]:
    """Random subsample so no label has more than `cap` samples (deterministic for a seed)."""
    raise NotImplementedError


def write_manifest(samples: list[Sample], path: Path) -> None:
    """JSONL, one dataclasses.asdict(sample) per line."""
    raise NotImplementedError


def read_manifest(path: Path) -> list[Sample]:
    raise NotImplementedError


def read_exif_gps(path: str) -> tuple[float, float] | None:
    """Decimal (lat, lon) from EXIF GPS tags via Pillow, or None."""
    raise NotImplementedError
