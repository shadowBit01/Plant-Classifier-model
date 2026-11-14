"""Lane L1 (data-eval). Dataset loaders, grouped splits, manifests. Tests: tests/test_eval.py

Expected raw layouts under DATA_DIR (inspect with `find $VANALENS_DATA/<name> -maxdepth 2 | head`):
  inat_india/     <Genus species>/<obs_id>.jpg + meta.csv (image,lat,lon,obs_id,observed_on,license,url);
                  built automatically by card L1-b from the iNaturalist API (primary evaluation set)
  dimpsar/        optional; as downloaded from Mendeley (10.17632/748f8jkphb), folder per species
  plantnet300k/   optional (disabled); images/{train,val,test}/<species_id>/*.jpg + plantnet300K_species_id_2_name.json
  campus/         optional; <Genus species>/*.jpg, optional meta.csv (image,lat,lon,organ,plant_id)
Every loader returns (samples, taxa) with taxa idx 0..C-1 sorted by species name, and writes
DATA_DIR/<name>/labels.json via contracts.save_taxa. Genus = first word of species; family from a
lookup (GBIF species/match, cached) or "" if unknown.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

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
    rng = np.random.default_rng(seed)
    group_to_samples: dict[str, list[Sample]] = {}
    for s in samples:
        g = s.group if s.group else s.image_id
        group_to_samples.setdefault(g, []).append(s)

    label_to_groups: dict[int | None, list[str]] = {}
    for g, grp_samples in group_to_samples.items():
        lbl = grp_samples[0].label
        label_to_groups.setdefault(lbl, []).append(g)

    splits: dict[str, list[Sample]] = {"gallery": [], "cal": [], "test": []}
    f_gal, f_cal, f_test = fractions
    total_f = f_gal + f_cal + f_test
    f_gal, f_cal, f_test = f_gal / total_f, f_cal / total_f, f_test / total_f

    sorted_labels = sorted(label_to_groups.keys(), key=lambda x: (x is None, -1 if x is None else x))
    for lbl in sorted_labels:
        groups = label_to_groups[lbl]
        perm = rng.permutation(len(groups))
        shuffled = [groups[int(i)] for i in perm]
        m = len(shuffled)
        if m >= 3:
            n_cal = max(1, int(round(m * f_cal)))
            n_test = max(1, int(round(m * f_test)))
            while n_cal + n_test >= m:
                if n_cal >= n_test and n_cal > 1:
                    n_cal -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
            n_gal = m - n_cal - n_test
        elif m == 2:
            n_gal, n_cal, n_test = 1, 0, 1
        elif m == 1:
            n_gal, n_cal, n_test = 1, 0, 0
        else:
            n_gal, n_cal, n_test = 0, 0, 0

        for g in shuffled[:n_gal]:
            splits["gallery"].extend(group_to_samples[g])
        for g in shuffled[n_gal:n_gal + n_cal]:
            splits["cal"].extend(group_to_samples[g])
        for g in shuffled[n_gal + n_cal:]:
            splits["test"].extend(group_to_samples[g])

    return splits


def cap_per_class(samples: list[Sample], cap: int = 100, seed: int = 0) -> list[Sample]:
    """Random subsample so no label has more than `cap` samples (deterministic for a seed)."""
    rng = np.random.default_rng(seed)
    by_label: dict[int | None, list[int]] = {}
    for idx, s in enumerate(samples):
        by_label.setdefault(s.label, []).append(idx)

    kept_indices: list[int] = []
    sorted_labels = sorted(by_label.keys(), key=lambda x: (x is None, -1 if x is None else x))
    for lbl in sorted_labels:
        idxs = by_label[lbl]
        if len(idxs) > cap:
            chosen = rng.choice(idxs, size=cap, replace=False)
            kept_indices.extend(int(i) for i in chosen)
        else:
            kept_indices.extend(idxs)

    kept_indices.sort()
    return [samples[i] for i in kept_indices]


def write_manifest(samples: list[Sample], path: Path) -> None:
    """JSONL, one dataclasses.asdict(sample) per line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(s)) for s in samples]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_manifest(path: Path) -> list[Sample]:
    p = Path(path)
    return [
        Sample(**json.loads(line))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rational_to_float(val: object) -> float:
    if isinstance(val, tuple) and len(val) == 2:
        return float(val[0]) / float(val[1])
    return float(val)  # type: ignore[arg-type]


def _dms_to_deg(dms: object) -> float:
    if isinstance(dms, (tuple, list)) and len(dms) == 3:
        d, m, s = (_rational_to_float(x) for x in dms)
        return d + m / 60.0 + s / 3600.0
    return _rational_to_float(dms)


def read_exif_gps(path: str) -> tuple[float, float] | None:
    """Decimal (lat, lon) from EXIF GPS tags via Pillow, or None."""
    try:
        with Image.open(path) as img:
            gps_info: dict | None = None
            if hasattr(img, "getexif"):
                exif = img.getexif()
                if exif:
                    try:
                        gps_info = exif.get_ifd(0x8825)
                    except Exception:
                        gps_info = None
            if not gps_info and hasattr(img, "_getexif"):
                raw_exif = img._getexif()  # type: ignore[attr-defined]
                if isinstance(raw_exif, dict):
                    gps_info = raw_exif.get(0x8825)
            if not gps_info or 2 not in gps_info or 4 not in gps_info:
                return None

            lat = _dms_to_deg(gps_info[2])
            lon = _dms_to_deg(gps_info[4])
            lat_ref = gps_info.get(1, "N")
            lon_ref = gps_info.get(3, "E")
            if isinstance(lat_ref, bytes):
                lat_ref = lat_ref.decode("ascii", errors="ignore")
            if isinstance(lon_ref, bytes):
                lon_ref = lon_ref.decode("ascii", errors="ignore")
            if str(lat_ref).strip().upper().startswith("S"):
                lat = -abs(lat)
            if str(lon_ref).strip().upper().startswith("W"):
                lon = -abs(lon)
            if not (np.isfinite(lat) and np.isfinite(lon)):
                return None
            return (float(lat), float(lon))
    except Exception:
        return None

