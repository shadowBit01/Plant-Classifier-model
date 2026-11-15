"""Build evaluation dataset manifests under $VANALENS_CACHE/manifests/ and labels.json under $VANALENS_DATA/<dataset>/."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vanalens.config import CACHE_DIR, DATA_DIR, GALLERY_CAP
from vanalens.contracts import Taxon, load_taxa, save_taxa
from vanalens.data import (
    build_inat_india,
    cap_per_class,
    grouped_split,
    load_campus,
    load_dimpsar,
    load_inat_india,
    load_vl600,
    write_manifest,
)


def _process_inat_india(root: Path, manifests_dir: Path) -> None:
    if not (root / "meta.csv").exists():
        build_inat_india(root)

    samples, taxa = load_inat_india(root)
    save_taxa(taxa, root / "labels.json")

    obs_dates: dict[str, str] = {}
    with (root / "meta.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            oid = str(row.get("obs_id") or "").strip()
            dt = str(row.get("observed_on") or "").strip()[:10]
            if oid and dt:
                obs_dates[oid] = dt

    pre_samples = [s for s in samples if obs_dates.get(s.group, "") < "2025-06-01"]
    test_samples = [s for s in samples if obs_dates.get(s.group, "") >= "2025-07-01"]

    pre_splits = grouped_split(pre_samples, fractions=(0.8, 0.2, 0.0), seed=0)
    gallery = cap_per_class(pre_splits["gallery"], cap=GALLERY_CAP, seed=0)
    cal = pre_splits["cal"]

    write_manifest(gallery, manifests_dir / "inat_india_gallery.jsonl")
    write_manifest(cal, manifests_dir / "inat_india_cal.jsonl")
    write_manifest(test_samples, manifests_dir / "inat_india_test.jsonl")

    print(
        f"inat_india: classes={len(taxa)} gallery={len(gallery)} cal={len(cal)} test={len(test_samples)}"
    )


def _process_vl600(root: Path, manifests_dir: Path) -> dict[str, object]:
    samples, taxa = load_vl600(root)
    save_taxa(taxa, root / "labels.json")

    obs_dates: dict[str, str] = {}
    meta_path = root / "meta.csv"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                oid = str(row.get("obs_id") or "").strip()
                dt = str(row.get("observed_on") or "").strip()[:10]
                if oid and dt:
                    obs_dates[oid] = dt

    pre_samples = [s for s in samples if obs_dates.get(s.group, "") < "2025-07-01"]
    test_samples = [s for s in samples if obs_dates.get(s.group, "") >= "2025-07-01"]

    pre_splits = grouped_split(pre_samples, fractions=(0.8, 0.2, 0.0), seed=0)
    gallery = cap_per_class(pre_splits["gallery"], cap=GALLERY_CAP, seed=0)
    cal = pre_splits["cal"]

    manifests_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(gallery, manifests_dir / "vl600_gallery.jsonl")
    write_manifest(cal, manifests_dir / "vl600_cal.jsonl")
    write_manifest(test_samples, manifests_dir / "vl600_test.jsonl")

    test_obs_by_label: dict[int, set[str]] = {t.idx: set() for t in taxa}
    for s in test_samples:
        if s.label is not None and s.label in test_obs_by_label:
            test_obs_by_label[s.label].add(s.group)

    lt5_species = [t.species for t in taxa if len(test_obs_by_label[t.idx]) < 5]
    gal_obs = len({s.group for s in gallery})
    cal_obs = len({s.group for s in cal})
    test_obs = len({s.group for s in test_samples})

    print(
        f"vl600: classes={len(taxa)} gallery={len(gallery)} cal={len(cal)} test={len(test_samples)} "
        f"lt5_test_obs={len(lt5_species)}"
    )
    return {
        "taxa": taxa,
        "species": len(taxa),
        "photos_per_split": {
            "gallery": len(gallery),
            "cal": len(cal),
            "test": len(test_samples),
            "pre_uncapped": len(pre_splits["gallery"]),
        },
        "observations_per_split": {
            "gallery": gal_obs,
            "cal": cal_obs,
            "test": test_obs,
            "total": len({s.group for s in samples}),
        },
        "species_lt5_test_obs": len(lt5_species),
        "species_lt5_list": lt5_species,
    }


def _process_campus(
    root: Path, manifests_dir: Path, vl600_taxa: list[Taxon] | None = None
) -> dict[str, object]:
    if vl600_taxa is None and (DATA_DIR / "vl600" / "labels.json").exists():
        vl600_taxa = load_taxa(DATA_DIR / "vl600" / "labels.json")

    samples, taxa = load_campus(root, taxa=vl600_taxa)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(samples, manifests_dir / "campus_test.jsonl")

    open_set_species = sorted(
        {
            s.image_id.split("/")[1]
            for s in samples
            if s.label is None and "/" in s.image_id
        }
    )
    print(
        f"campus: test={len(samples)} in_vl600={sum(1 for s in samples if s.label is not None)} "
        f"open_set_species={len(open_set_species)}"
    )
    return {
        "test": len(samples),
        "in_vl600": sum(1 for s in samples if s.label is not None),
        "open_set_species": open_set_species,
        "taxa": taxa,
    }


def _process_standard_dataset(name: str, loader, root: Path, manifests_dir: Path) -> None:
    try:
        samples, taxa = loader(root)
    except NotImplementedError:
        return
    save_taxa(taxa, root / "labels.json")
    splits = grouped_split(samples, fractions=(0.6, 0.2, 0.2), seed=0)
    gallery = cap_per_class(splits["gallery"], cap=GALLERY_CAP, seed=0)
    cal = splits["cal"]
    test = splits["test"]

    write_manifest(gallery, manifests_dir / f"{name}_gallery.jsonl")
    write_manifest(cal, manifests_dir / f"{name}_cal.jsonl")
    write_manifest(test, manifests_dir / f"{name}_test.jsonl")

    print(
        f"{name}: classes={len(taxa)} gallery={len(gallery)} cal={len(cal)} test={len(test)}"
    )


def main() -> None:
    manifests_dir = CACHE_DIR / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    if (DATA_DIR / "inat_india").exists():
        _process_inat_india(DATA_DIR / "inat_india", manifests_dir)

    vl600_taxa: list[Taxon] | None = None
    if (DATA_DIR / "vl600").exists():
        res = _process_vl600(DATA_DIR / "vl600", manifests_dir)
        vl600_taxa = res["taxa"]  # type: ignore[assignment]

    if (DATA_DIR / "dimpsar").exists():
        _process_standard_dataset("dimpsar", load_dimpsar, DATA_DIR / "dimpsar", manifests_dir)

    if (DATA_DIR / "campus").exists():
        _process_campus(DATA_DIR / "campus", manifests_dir, vl600_taxa=vl600_taxa)


if __name__ == "__main__":
    main()

