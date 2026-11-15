"""Build evaluation dataset manifests under $VANALENS_CACHE/manifests/ and labels.json under $VANALENS_DATA/<dataset>/."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vanalens.config import CACHE_DIR, DATA_DIR, GALLERY_CAP
from vanalens.contracts import save_taxa
from vanalens.data import (
    build_inat_india,
    cap_per_class,
    grouped_split,
    load_campus,
    load_dimpsar,
    load_inat_india,
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

    _process_inat_india(DATA_DIR / "inat_india", manifests_dir)

    if (DATA_DIR / "dimpsar").exists():
        _process_standard_dataset("dimpsar", load_dimpsar, DATA_DIR / "dimpsar", manifests_dir)

    if (DATA_DIR / "campus").exists():
        _process_standard_dataset("campus", load_campus, DATA_DIR / "campus", manifests_dir)


if __name__ == "__main__":
    main()
