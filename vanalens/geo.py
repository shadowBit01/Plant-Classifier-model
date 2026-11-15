"""Lane L4 (context). Location prior from GBIF occurrence counts. Tests: tests/test_context.py

GBIF occurrence search (no key needed):
  https://api.gbif.org/v1/occurrence/search?geoDistance={lat},{lon},{radius}km
      &kingdomKey=6&facet=speciesKey&facetLimit=10000&limit=0
-> response["facets"][0]["counts"] = [{"name": "<speciesKey>", "count": n}, ...]
Verify the exact format against https://techdocs.gbif.org/en/openapi/ before relying on it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contracts import Taxon


def gbif_species_counts(lat: float, lon: float, radius_km: int = 50, kingdom_key: int = 6,
                        cache_dir: Path | None = None) -> dict[int, int]:
    """{gbif speciesKey: occurrence count} around (lat, lon). Cache JSON per cell rounded to 0.5 degree
    under CACHE_DIR/gbif/. Timeout 10 s; on failure return {} (caller then skips the prior)."""
    import urllib.request

    from .config import CACHE_DIR

    if cache_dir is None:
        cache_dir = CACHE_DIR / "gbif"

    # Round to 0.5-degree grid for caching
    cell_lat = round(lat * 2) / 2
    cell_lon = round(lon * 2) / 2
    cache_file = Path(cache_dir) / f"{cell_lat}_{cell_lon}_{radius_km}_{kingdom_key}.json"

    if cache_file.exists():
        try:
            return {int(k): v for k, v in json.loads(cache_file.read_text()).items()}
        except Exception:
            pass

    url = (
        f"https://api.gbif.org/v1/occurrence/search"
        f"?geoDistance={lat},{lon},{radius_km}km"
        f"&kingdomKey={kingdom_key}&facet=speciesKey&facetLimit=10000&limit=0"
    )
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "VanaLens/1.0 (plant-classifier)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        facets = data.get("facets", [])
        counts: dict[int, int] = {}
        if facets:
            for entry in facets[0].get("counts", []):
                counts[int(entry["name"])] = int(entry["count"])
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(counts))
        return counts
    except Exception:
        return {}


def resolve_gbif_keys(taxa: list[Taxon], cache_dir: Path | None = None) -> list[Taxon]:
    """Fill Taxon.gbif_key via https://api.gbif.org/v1/species/match?name=<species>&kingdom=Plantae
    (use usageKey, or speciesKey when present). Cache results. Leave None when no confident match."""
    import urllib.request
    import urllib.parse

    from .config import CACHE_DIR

    if cache_dir is None:
        cache_dir = CACHE_DIR / "gbif"

    cache_file = Path(cache_dir) / "species_keys.json"
    cache: dict[str, int | None] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except Exception:
            pass

    for taxon in taxa:
        if taxon.gbif_key is not None:
            continue
        if taxon.species in cache:
            taxon.gbif_key = cache[taxon.species]
            continue
        params = urllib.parse.urlencode({"name": taxon.species, "kingdom": "Plantae"})
        url = f"https://api.gbif.org/v1/species/match?{params}"
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": "VanaLens/1.0 (plant-classifier)"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            if data.get("matchType", "NONE") != "NONE":
                key = data.get("speciesKey") or data.get("usageKey")
                taxon.gbif_key = int(key) if key is not None else None
            else:
                taxon.gbif_key = None
        except Exception:
            taxon.gbif_key = None
        cache[taxon.species] = taxon.gbif_key

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache))
    return taxa


def geo_prior(counts: dict[int, int], taxa: list[Taxon], smoothing: float = 1.0) -> np.ndarray:
    """(C,) prior: (counts.get(gbif_key, 0) + smoothing) normalised to sum 1.
    Taxa without gbif_key get just `smoothing`. Empty counts -> uniform."""
    C = len(taxa)
    if not counts:
        return np.full(C, 1.0 / C)

    raw = np.array(
        [counts.get(t.gbif_key, 0) + smoothing if t.gbif_key is not None else smoothing
         for t in taxa],
        dtype=np.float64,
    )
    return (raw / raw.sum()).astype(np.float64)


def apply_prior(P: np.ndarray, prior: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """Row-wise P * prior**beta, renormalised. beta = 0 must return P unchanged (up to float error)."""
    weighted = P * (prior ** beta)
    total = weighted.sum(axis=1, keepdims=True)
    return weighted / total


def prefetch_gbif_cells(
    coords: list[tuple[float, float]],
    radius_km: int = 50,
    kingdom_key: int = 6,
    cache_dir: Path | None = None,
    max_workers: int = 10,
    retries: int = 4,
) -> int:
    """Pre-fetch and cache GBIF occurrence counts for a list of (lat, lon) coordinates in parallel.
    Skips coordinates whose 0.5-degree grid cell is already cached.
    Retries uncached cells across multiple passes.
    Returns total count of unique cells cached."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .config import CACHE_DIR

    if cache_dir is None:
        cache_dir = CACHE_DIR / "gbif"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    unique_cells = set()
    for lat, lon in coords:
        if lat is not None and lon is not None:
            cell_lat = round(lat * 2) / 2
            cell_lon = round(lon * 2) / 2
            unique_cells.add((cell_lat, cell_lon))

    for attempt in range(retries):
        to_fetch = [
            (clat, clon)
            for (clat, clon) in unique_cells
            if not (cache_dir / f"{clat}_{clon}_{radius_km}_{kingdom_key}.json").exists()
        ]
        if not to_fetch:
            break

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    gbif_species_counts,
                    lat=clat,
                    lon=clon,
                    radius_km=radius_km,
                    kingdom_key=kingdom_key,
                    cache_dir=cache_dir,
                )
                for clat, clon in to_fetch
            ]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    pass
        time.sleep(1)

    cached_count = sum(
        1 for (clat, clon) in unique_cells
        if (cache_dir / f"{clat}_{clon}_{radius_km}_{kingdom_key}.json").exists()
    )
    return cached_count


