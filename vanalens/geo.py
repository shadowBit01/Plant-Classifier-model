"""Lane L4 (context). Location prior from GBIF occurrence counts. Tests: tests/test_context.py

GBIF occurrence search (no key needed):
  https://api.gbif.org/v1/occurrence/search?geoDistance={lat},{lon},{radius}km
      &kingdomKey=6&facet=speciesKey&facetLimit=10000&limit=0
-> response["facets"][0]["counts"] = [{"name": "<speciesKey>", "count": n}, ...]
Verify the exact format against https://techdocs.gbif.org/en/openapi/ before relying on it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import Taxon


def gbif_species_counts(lat: float, lon: float, radius_km: int = 50, kingdom_key: int = 6,
                        cache_dir: Path | None = None) -> dict[int, int]:
    """{gbif speciesKey: occurrence count} around (lat, lon). Cache JSON per cell rounded to 0.5 degree
    under CACHE_DIR/gbif/. Timeout 10 s; on failure return {} (caller then skips the prior)."""
    raise NotImplementedError


def resolve_gbif_keys(taxa: list[Taxon], cache_dir: Path | None = None) -> list[Taxon]:
    """Fill Taxon.gbif_key via https://api.gbif.org/v1/species/match?name=<species>&kingdom=Plantae
    (use usageKey, or speciesKey when present). Cache results. Leave None when no confident match."""
    raise NotImplementedError


def geo_prior(counts: dict[int, int], taxa: list[Taxon], smoothing: float = 1.0) -> np.ndarray:
    """(C,) prior: (counts.get(gbif_key, 0) + smoothing) normalised to sum 1.
    Taxa without gbif_key get just `smoothing`. Empty counts -> uniform."""
    raise NotImplementedError


def apply_prior(P: np.ndarray, prior: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """Row-wise P * prior**beta, renormalised. beta = 0 must return P unchanged (up to float error)."""
    raise NotImplementedError
