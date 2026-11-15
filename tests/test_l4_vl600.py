"""Lane L4 tests: offline tests for vl600 text prototypes, traits, fallback, and GBIF cache."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vanalens import config, contracts
from vanalens.geo import geo_prior
from vanalens.scorers import TextScorer
from vanalens.traits import taxonomic_prompt


def test_vl600_text_protos_exists_and_format():
    npz_path = config.CACHE_DIR / "text" / "vl600.npz"
    assert npz_path.exists(), f"Missing text prototypes at {npz_path}"

    data = np.load(npz_path, allow_pickle=False)
    required_keys = {"stacked_protos", "counts", "species", "strings_json", "traits_json"}
    for key in required_keys:
        assert key in data, f"Missing key {key} in vl600.npz"

    species = [str(s) for s in data["species"]]
    labels_path = config.DATA_DIR / "vl600" / "labels.json"
    taxa = contracts.load_taxa(labels_path)

    assert len(species) == len(taxa) == 607, f"Expected 607 species, got {len(species)}"
    assert species == [t.species for t in taxa], "Prototypes species do not match labels.json order"

    stacked = data["stacked_protos"]
    counts = data["counts"]
    assert stacked.ndim == 2 and stacked.shape[1] == config.BIOCLIP_DIM
    assert stacked.dtype == np.float32
    assert counts.shape == (607,)
    assert counts.sum() == stacked.shape[0]

    # All vectors must be L2-normalised
    norms = np.linalg.norm(stacked, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "Every prototype vector must be L2-normalised"


def test_traits_and_taxonomic_fallback():
    npz_path = config.CACHE_DIR / "text" / "vl600.npz"
    data = np.load(npz_path, allow_pickle=False)

    counts = data["counts"]
    strings = json.loads(str(data["strings_json"]))
    traits = json.loads(str(data["traits_json"]))

    species_with_traits = [s for s, t in traits.items() if t]
    fallback_species = [s for s, t in traits.items() if not t]

    assert len(species_with_traits) + len(fallback_species) == 607
    assert len(species_with_traits) >= 490, "Expected at least 490 species with traits"
    assert len(fallback_species) > 0, "Expected taxonomic fallback species"

    for i, count in enumerate(counts):
        s_list = strings[i]
        assert len(s_list) == count
        if count == 7:
            # 6 traits + 1 species name prompt
            assert s_list[-1].startswith("a photo of ")
        elif count == 1:
            # 1 taxonomic prompt
            assert s_list[0].startswith("a photo of ")
        else:
            pytest.fail(f"Unexpected prompt count {count} for species index {i}")


def test_taxonomic_prompt_formatting():
    t = contracts.Taxon(idx=0, species="Ficus religiosa", genus="Ficus", family="Moraceae")
    p1 = taxonomic_prompt(t, "sacred fig")
    assert p1 == "a photo of Moraceae Ficus Ficus religiosa, sacred fig"

    p2 = taxonomic_prompt(t, None)
    assert p2 == "a photo of Moraceae Ficus Ficus religiosa"


def test_gbif_cells_cached_for_vl600():
    manifest_path = config.CACHE_DIR / "manifests" / "vl600_test.jsonl"
    assert manifest_path.exists(), f"Test manifest missing at {manifest_path}"

    needed_cells = set()
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            lat = sample.get("lat")
            lon = sample.get("lon")
            if lat is not None and lon is not None:
                needed_cells.add((round(lat * 2) / 2, round(lon * 2) / 2))

    assert len(needed_cells) == 1622, f"Expected 1622 unique GPS cells, got {len(needed_cells)}"

    gbif_dir = config.CACHE_DIR / "gbif"
    assert gbif_dir.exists(), f"GBIF cache directory missing at {gbif_dir}"

    missing_files = []
    for clat, clon in needed_cells:
        cell_file = gbif_dir / f"{clat}_{clon}_50_6.json"
        if not cell_file.exists():
            missing_files.append(f"{clat}_{clon}")

    assert not missing_files, f"Missing {len(missing_files)} GBIF cache files"

    # Verify a sample cached file has valid JSON occurrence counts
    sample_cell = next(iter(needed_cells))
    sample_path = gbif_dir / f"{sample_cell[0]}_{sample_cell[1]}_50_6.json"
    counts = json.loads(sample_path.read_text())
    assert isinstance(counts, dict)


def test_offline_geo_prior_with_vl600_cells():
    labels_path = config.DATA_DIR / "vl600" / "labels.json"
    taxa = contracts.load_taxa(labels_path)

    gbif_dir = config.CACHE_DIR / "gbif"
    sample_path = next(gbif_dir.glob("*_50_6.json"))
    counts = {int(k): v for k, v in json.loads(sample_path.read_text()).items()}

    prior = geo_prior(counts, taxa)
    assert prior.shape == (len(taxa),)
    assert np.all(prior >= 0)
    assert np.allclose(prior.sum(), 1.0, atol=1e-6)


def test_text_scorer_offline_vl600():
    npz_path = config.CACHE_DIR / "text" / "vl600.npz"
    d = np.load(npz_path, allow_pickle=False)

    counts = d["counts"].astype(int)
    flat = d["stacked_protos"].reshape(-1, d["stacked_protos"].shape[-1]).astype(np.float64)
    strings = json.loads(str(d["strings_json"]))
    offs = np.concatenate([[0], np.cumsum(counts)])
    protos = [flat[offs[i] : offs[i + 1]] for i in range(len(counts))]

    scorer = TextScorer(protos=protos, trait_strings=strings)

    # Fake query embeddings (4 samples, 1792-d [bioclip | dino])
    rng = np.random.default_rng(42)
    fake_X = rng.normal(size=(4, config.EMB_DIM)).astype(np.float32)

    P = scorer.predict_proba(fake_X)
    contracts.check_proba(P, len(counts))

    # Test explain returns valid strings
    explanation = scorer.explain(fake_X[0], c=0, top=3)
    assert len(explanation) > 0
    assert all(isinstance(s, str) for s in explanation)
