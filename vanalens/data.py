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

from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

from .config import CACHE_DIR, DATA_DIR
from .contracts import Sample, Taxon, load_taxa, save_taxa

INAT_API = "https://api.inaturalist.org/v1"
INAT_USER_AGENT = "VanaLens/1.0 (https://github.com/Plant-Classifier-model; research evaluation)"
ALLOWED_LICENSES = {"cc0", "cc-by", "cc-by-nc"}
_LAST_INAT_CALL = 0.0


def _inat_get(endpoint: str, params: dict[str, object]) -> dict:
    """Rate-limited GET against api.inaturalist.org (max 1 req/s), cached under CACHE_DIR/inat_api."""
    global _LAST_INAT_CALL
    import hashlib

    query = urllib.parse.urlencode(sorted(params.items()))
    url = f"{INAT_API}/{endpoint.lstrip('/')}?{query}"
    cache_dir = CACHE_DIR / "inat_api"
    cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    for attempt in range(4):
        wait = 1.02 - (time.monotonic() - _LAST_INAT_CALL)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": INAT_USER_AGENT, "Accept": "application/json"},
        )
        _LAST_INAT_CALL = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(data), encoding="utf-8")
            except Exception:
                pass
            return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))
    return {}


def _parse_obs_coords(obs: dict) -> tuple[float, float] | None:
    geojson = obs.get("geojson")
    if isinstance(geojson, dict):
        coords = geojson.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            try:
                lon, lat = float(coords[0]), float(coords[1])
                if np.isfinite(lat) and np.isfinite(lon):
                    return lat, lon
            except Exception:
                pass
    loc = obs.get("location")
    if isinstance(loc, str) and "," in loc:
        try:
            lat_s, lon_s = loc.split(",", 1)
            lat, lon = float(lat_s.strip()), float(lon_s.strip())
            if np.isfinite(lat) and np.isfinite(lon):
                return lat, lon
        except Exception:
            pass
    return None


def _medium_photo_from_obs(obs: dict) -> tuple[str, str] | None:
    """Return (medium_url, license_code) for the first CC0/CC-BY/CC-BY-NC photo in `obs`."""
    for p in obs.get("photos") or []:
        if not isinstance(p, dict):
            continue
        lic = str(p.get("license_code") or "").strip().lower()
        if lic not in ALLOWED_LICENSES:
            continue
        raw_url = str(p.get("url") or p.get("medium_url") or "").strip()
        if not raw_url:
            continue
        med_url = re.sub(r"/(square|small|thumb)\.", "/medium.", raw_url)
        return med_url, lic
    return None


def _fetch_species_obs(
    place_id: int,
    taxon_id: int,
    *,
    d1: str | None = None,
    d2: str | None = None,
    limit: int = 30,
) -> list[dict[str, object]]:
    params: dict[str, object] = {
        "place_id": place_id,
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "geo": "true",
        "photos": "true",
        "photo_license": "cc0,cc-by,cc-by-nc",
        "per_page": limit + 10,
        "order_by": "id",
        "order": "desc",
    }
    if d1 is not None:
        params["d1"] = d1
    if d2 is not None:
        params["d2"] = d2

    data = _inat_get("observations", params)
    out: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for obs in data.get("results") or []:
        if not isinstance(obs, dict) or "id" not in obs:
            continue
        obs_id = str(obs["id"])
        if obs_id in seen_ids:
            continue
        observed_on = str(
            obs.get("observed_on")
            or (obs.get("observed_on_details") or {}).get("date")
            or ""
        ).strip()[:10]
        if len(observed_on) != 10:
            continue
        if d2 is not None and not (observed_on < "2025-06-01"):
            continue
        if d1 is not None and not (observed_on >= "2025-07-01"):
            continue
        coords = _parse_obs_coords(obs)
        if coords is None:
            continue
        photo_info = _medium_photo_from_obs(obs)
        if photo_info is None:
            continue
        lat, lon = coords
        med_url, lic = photo_info
        seen_ids.add(obs_id)
        out.append(
            {
                "obs_id": obs_id,
                "observed_on": observed_on,
                "lat": lat,
                "lon": lon,
                "license": lic,
                "url": med_url,
            }
        )
        if len(out) >= limit:
            break
    return out


def _download_photo(task: tuple[Path, str]) -> bool:
    dest, url = task
    if dest.exists() and dest.stat().st_size > 100:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": INAT_USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read()
            with Image.open(BytesIO(raw)) as im:
                rgb = im.convert("RGB")
                rgb.save(dest, format="JPEG", quality=92)
            return True
        except Exception:
            if attempt == 2:
                return False
            time.sleep(1.0 * (attempt + 1))
    return False


def build_inat_india(root: Path = DATA_DIR / "inat_india") -> Path:
    """Build $VANALENS_DATA/inat_india automatically from the iNaturalist API v1."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    places = _inat_get("places/autocomplete", {"q": "Madhya Pradesh"}).get("results") or []
    place_id = 7745
    for p in places:
        if isinstance(p, dict) and (
            p.get("name") == "Madhya Pradesh"
            or "Madhya Pradesh" in str(p.get("display_name") or "")
        ):
            place_id = int(p["id"])
            break

    sp_counts = _inat_get(
        "observations/species_counts",
        {
            "place_id": place_id,
            "taxon_id": 47126,
            "quality_grade": "research",
            "hrank": "species",
            "lrank": "species",
            "per_page": 60,
        },
    ).get("results") or []

    species_records: list[tuple[str, list[dict[str, object]]]] = []
    for entry in sp_counts[:60]:
        taxon = entry.get("taxon") or {}
        taxon_id = int(taxon["id"])
        sp_name = str(taxon["name"]).strip()
        if len(sp_name.split()) < 2:
            continue
        sp_name = " ".join(sp_name.split()[:2])

        test_obs = _fetch_species_obs(place_id, taxon_id, d1="2025-07-01", limit=15)
        if len(test_obs) < 5:
            continue
        pre_obs = _fetch_species_obs(place_id, taxon_id, d2="2025-05-31", limit=30)
        seen: set[str] = set()
        combined: list[dict[str, object]] = []
        for rec in pre_obs + test_obs:
            oid = str(rec["obs_id"])
            if oid not in seen:
                seen.add(oid)
                combined.append(rec)
        species_records.append((sp_name, combined))

    download_tasks: list[tuple[Path, str]] = []
    for sp_name, recs in species_records:
        for rec in recs:
            dest = root / sp_name / f"{rec['obs_id']}.jpg"
            download_tasks.append((dest, str(rec["url"])))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_download_photo, download_tasks))

    ok_paths = {task[0] for task, ok in zip(download_tasks, results) if ok}

    meta_rows: list[dict[str, object]] = []
    for sp_name, recs in sorted(species_records, key=lambda x: x[0]):
        kept_recs = [
            r for r in recs if (root / sp_name / f"{r['obs_id']}.jpg") in ok_paths
        ]
        n_test = sum(1 for r in kept_recs if str(r["observed_on"]) >= "2025-07-01")
        if n_test < 5:
            continue
        for r in kept_recs:
            meta_rows.append(
                {
                    "image": f"{sp_name}/{r['obs_id']}.jpg",
                    "lat": r["lat"],
                    "lon": r["lon"],
                    "obs_id": r["obs_id"],
                    "observed_on": r["observed_on"],
                    "license": r["license"],
                    "url": r["url"],
                }
            )

    meta_path = root / "meta.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "lat", "lon", "obs_id", "observed_on", "license", "url"],
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    return root


def _lookup_gbif_taxa_info(
    species_names: list[str],
    existing_labels: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[str, tuple[str, int | None]]:
    """Return {species: (family, gbif_key)} using cached GBIF species/match lookups."""
    if cache_dir is None:
        cache_dir = CACHE_DIR / "gbif"
    cache_file = Path(cache_dir) / "species_match.json"
    cache: dict[str, dict[str, object]] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    if existing_labels is not None and existing_labels.exists():
        try:
            for t in load_taxa(existing_labels):
                if t.species not in cache and (t.family or t.gbif_key is not None):
                    cache[t.species] = {"family": t.family, "gbif_key": t.gbif_key}
        except Exception:
            pass

    updated = False
    out: dict[str, tuple[str, int | None]] = {}
    missing: list[str] = []
    for sp in species_names:
        if sp in cache:
            entry = cache[sp]
            fam = str(entry.get("family") or "")
            key = entry.get("gbif_key")
            out[sp] = (fam, int(key) if key is not None else None)
        else:
            missing.append(sp)

    if missing:
        import os

        offline = os.environ.get("VANALENS_OFFLINE", "0") == "1"

        def _fetch_one(sp: str) -> tuple[str, str, int | None]:
            if offline:
                return sp, "", None
            fam = ""
            gbif_key: int | None = None
            params = urllib.parse.urlencode({"name": sp, "kingdom": "Plantae"})
            url = f"https://api.gbif.org/v1/species/match?{params}"
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                if data.get("matchType", "NONE") != "NONE":
                    fam = str(data.get("family") or "")
                    raw_key = data.get("speciesKey") or data.get("usageKey")
                    gbif_key = int(raw_key) if raw_key is not None else None
            except Exception:
                pass
            return sp, fam, gbif_key

        if len(missing) > 4 and not offline:
            with ThreadPoolExecutor(max_workers=8) as pool:
                fetched = list(pool.map(_fetch_one, missing))
        else:
            fetched = [_fetch_one(sp) for sp in missing]

        for sp, fam, gbif_key in fetched:
            cache[sp] = {"family": fam, "gbif_key": gbif_key}
            out[sp] = (fam, gbif_key)
        if not offline:
            updated = True

    if updated:
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        except Exception:
            pass

    return out


def load_inat_india(root: Path = DATA_DIR / "inat_india") -> tuple[list[Sample], list[Taxon]]:
    """Load inat_india dataset from `root` (source="inat", group=obs_id, lat/lon filled)."""
    root = Path(root)
    meta_by_img: dict[str, dict[str, str]] = {}
    meta_by_obs: dict[str, dict[str, str]] = {}
    meta_path = root / "meta.csv"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img_rel = str(row.get("image") or "").replace("\\", "/").strip()
                obs_id = str(row.get("obs_id") or "").strip()
                if img_rel:
                    meta_by_img[img_rel] = row
                if obs_id:
                    meta_by_obs[obs_id] = row

    if meta_by_img:
        species_set: set[str] = set()
        for img_rel in meta_by_img:
            if "/" in img_rel and (root / img_rel).exists():
                species_set.add(img_rel.split("/", 1)[0])
        species_names = sorted(species_set)
    else:
        species_names = sorted(
            {p.parent.name for p in root.glob("*/*.jpg") if p.parent != root}
        )

    gbif_info = _lookup_gbif_taxa_info(species_names, existing_labels=root / "labels.json")
    taxa = [
        Taxon(
            idx=i,
            species=sp,
            genus=sp.split()[0],
            family=gbif_info[sp][0],
            gbif_key=gbif_info[sp][1],
        )
        for i, sp in enumerate(species_names)
    ]
    save_taxa(taxa, root / "labels.json")

    sp_to_idx = {t.species: t.idx for t in taxa}
    samples: list[Sample] = []
    for sp in species_names:
        sp_dir = root / sp
        if not sp_dir.is_dir():
            continue
        for img_path in sorted(sp_dir.glob("*.jpg"), key=lambda p: p.stem):
            rel_img = f"{sp}/{img_path.name}"
            if meta_by_img and rel_img not in meta_by_img:
                continue
            obs_id = img_path.stem
            row = meta_by_img.get(rel_img) or meta_by_obs.get(obs_id) or {}
            group = str(row.get("obs_id") or obs_id).strip()
            lat: float | None = None
            lon: float | None = None
            try:
                if row.get("lat") not in (None, ""):
                    lat = float(row["lat"])
                if row.get("lon") not in (None, ""):
                    lon = float(row["lon"])
            except Exception:
                lat, lon = None, None
            samples.append(
                Sample(
                    image_id=f"inat_india/{sp}/{group}",
                    path=str(img_path.resolve()),
                    label=sp_to_idx[sp],
                    source="inat",
                    group=group,
                    organ=None,
                    lat=lat,
                    lon=lon,
                )
            )

    return samples, taxa


def load_vl600(root: Path = DATA_DIR / "vl600") -> tuple[list[Sample], list[Taxon]]:
    """Load vl600 dataset from `root` (source="inat", group=obs_id, lat/lon filled).
    Never drops a species from species.csv even if it has few or no photos/test observations."""
    root = Path(root)
    species_set: set[str] = set()
    species_csv = root / "species.csv"
    if species_csv.exists():
        with species_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sp = str(row.get("name") or row.get("species") or "").strip()
                if sp:
                    species_set.add(sp)

    meta_by_img: dict[str, dict[str, str]] = {}
    meta_path = root / "meta.csv"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img_rel = str(row.get("image") or "").replace("\\", "/").strip()
                if img_rel:
                    meta_by_img[img_rel] = row
                    if "/" in img_rel:
                        species_set.add(img_rel.split("/", 1)[0])

    if not species_set:
        species_set = {
            p.parent.name for p in root.glob("*/*.jpg") if p.parent != root
        }

    species_names = sorted(species_set)
    gbif_info = _lookup_gbif_taxa_info(species_names, existing_labels=root / "labels.json")
    taxa = [
        Taxon(
            idx=i,
            species=sp,
            genus=sp.split()[0],
            family=gbif_info[sp][0],
            gbif_key=gbif_info[sp][1],
        )
        for i, sp in enumerate(species_names)
    ]
    root.mkdir(parents=True, exist_ok=True)
    save_taxa(taxa, root / "labels.json")

    sp_to_idx = {t.species: t.idx for t in taxa}
    samples: list[Sample] = []
    for sp in species_names:
        sp_dir = root / sp
        if not sp_dir.is_dir():
            continue
        for img_path in sorted(sp_dir.glob("*.jpg"), key=lambda p: p.stem):
            rel_img = f"{sp}/{img_path.name}"
            if meta_by_img and rel_img not in meta_by_img:
                continue
            row = meta_by_img.get(rel_img) or {}
            stem = img_path.stem
            default_obs = stem.split("_", 1)[0] if "_" in stem else stem
            group = str(row.get("obs_id") or default_obs).strip()
            lat: float | None = None
            lon: float | None = None
            try:
                if row.get("lat") not in (None, ""):
                    lat_v = float(row["lat"])
                    if np.isfinite(lat_v):
                        lat = lat_v
                if row.get("lon") not in (None, ""):
                    lon_v = float(row["lon"])
                    if np.isfinite(lon_v):
                        lon = lon_v
            except Exception:
                lat, lon = None, None
            samples.append(
                Sample(
                    image_id=f"vl600/{sp}/{stem}",
                    path=str(img_path.resolve()),
                    label=sp_to_idx[sp],
                    source="inat",
                    group=group,
                    organ=None,
                    lat=lat,
                    lon=lon,
                )
            )

    return samples, taxa


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
    root = Path(root)
    meta_by_img: dict[str, dict[str, str]] = {}
    meta_path = root / "meta.csv"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img_rel = str(row.get("image") or "").replace("\\", "/").strip()
                if img_rel:
                    meta_by_img[img_rel] = row

    folder_names = sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    ) if root.exists() else []

    if taxa is None:
        gbif_info = _lookup_gbif_taxa_info(folder_names, existing_labels=root / "labels.json")
        out_taxa = [
            Taxon(
                idx=i,
                species=sp,
                genus=sp.split()[0],
                family=gbif_info[sp][0],
                gbif_key=gbif_info[sp][1],
            )
            for i, sp in enumerate(folder_names)
        ]
        if root.exists():
            save_taxa(out_taxa, root / "labels.json")
    else:
        out_taxa = list(taxa)

    sp_to_idx = {t.species: t.idx for t in out_taxa}
    samples: list[Sample] = []
    for sp in folder_names:
        sp_dir = root / sp
        img_files = sorted(
            [
                p
                for p in sp_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ],
            key=lambda p: p.name,
        )
        for img_path in img_files:
            rel_img = f"{sp}/{img_path.name}"
            row = meta_by_img.get(rel_img) or meta_by_img.get(img_path.name) or {}
            plant_id = str(row.get("plant_id") or "").strip()
            group = plant_id if plant_id else sp
            organ_raw = str(row.get("organ") or "").strip()
            organ = organ_raw if organ_raw else None

            lat: float | None = None
            lon: float | None = None
            try:
                if row.get("lat") not in (None, "") and row.get("lon") not in (None, ""):
                    lat_v = float(row["lat"])
                    lon_v = float(row["lon"])
                    if np.isfinite(lat_v) and np.isfinite(lon_v):
                        lat, lon = lat_v, lon_v
            except Exception:
                lat, lon = None, None
            if lat is None or lon is None:
                gps = read_exif_gps(str(img_path))
                if gps is not None:
                    lat, lon = gps

            samples.append(
                Sample(
                    image_id=f"campus/{sp}/{img_path.stem}",
                    path=str(img_path.resolve()),
                    label=sp_to_idx.get(sp, None),
                    source="campus",
                    group=group,
                    organ=organ,
                    lat=lat,
                    lon=lon,
                )
            )

    return samples, out_taxa


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
        if f_test == 0.0:
            n_test = 0
            if m >= 2 and f_cal > 0.0:
                n_cal = min(m - 1, max(1, int(round(m * f_cal))))
            else:
                n_cal = 0
            n_gal = m - n_cal
        elif m >= 3:
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

