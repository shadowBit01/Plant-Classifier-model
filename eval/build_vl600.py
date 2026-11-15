"""Build dataset vl600 automatically and resumably from iNaturalist API v1 and AWS open data S3.

Usage:
    python -m eval.build_vl600
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, timedelta
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from typing import Callable
import urllib.parse

import numpy as np
from PIL import Image
import requests
from requests.adapters import HTTPAdapter

from scripts.make_manifests import _process_campus, _process_vl600
from vanalens.config import CACHE_DIR, DATA_DIR, RUNS_DIR
from vanalens.data import ALLOWED_LICENSES, INAT_API, INAT_USER_AGENT

INDIA_PLACE_ID = 6681
PLANTAE_TAXON_ID = 47126
S3_BASE = "https://inaturalist-open-data.s3.amazonaws.com/photos"
EXPECTED_BYTES_PER_PHOTO = 80_000
MAX_EXPECTED_GB = 15.0
MIN_FREE_GB_AFTER = 20.0

# Common Indian list (c): (common_name, target_taxon_or_genus, is_genus)
COMMON_INDIAN_SPEC: list[tuple[str, str, bool]] = [
    ("tulsi", "Ocimum tenuiflorum", False),
    ("neem", "Azadirachta indica", False),
    ("hibiscus", "Hibiscus rosa-sinensis", False),
    ("rose", "Rosa", True),
    ("money plant", "Epipremnum aureum", False),
    ("bougainvillea", "Bougainvillea", True),
    ("marigold", "Tagetes erecta", False),
    ("mango", "Mangifera indica", False),
    ("peepal", "Ficus religiosa", False),
    ("banyan", "Ficus benghalensis", False),
    ("curry leaf", "Murraya koenigii", False),
    ("aloe vera", "Aloe vera", False),
    ("guava", "Psidium guajava", False),
    ("papaya", "Carica papaya", False),
    ("banana", "Musa", True),
    ("coconut", "Cocos nucifera", False),
    ("jasmine", "Jasminum sambac", False),
    ("oleander", "Nerium oleander", False),
    ("ixora", "Ixora coccinea", False),
    ("plumeria", "Plumeria rubra", False),
    ("gulmohar", "Delonix regia", False),
    ("amaltas", "Cassia fistula", False),
    ("ashoka", "Saraca asoca", False),
    ("false ashoka", "Monoon longifolium", False),
    ("teak", "Tectona grandis", False),
    ("moringa", "Moringa oleifera", False),
    ("tamarind", "Tamarindus indica", False),
    ("jamun", "Syzygium cumini", False),
    ("lantana", "Lantana camara", False),
    ("parthenium", "Parthenium hysterophorus", False),
    ("calotropis", "Calotropis procera", False),
    ("crown flower", "Calotropis gigantea", False),
    ("datura", "Datura", True),
    ("castor", "Ricinus communis", False),
    ("mimosa", "Mimosa pudica", False),
    ("snake plant", "Dracaena trifasciata", False),
    ("spider plant", "Chlorophytum comosum", False),
    ("crown of thorns", "Euphorbia milii", False),
    ("periwinkle", "Catharanthus roseus", False),
    ("croton", "Codiaeum variegatum", False),
    ("areca palm", "Dypsis lutescens", False),
    ("bamboo", "Bambusa vulgaris", False),
]

# Accepted iNaturalist binomial synonyms where iNat uses a different accepted name
INAT_SYNONYM_MAP: dict[str, str] = {
    "Murraya koenigii": "Bergera koenigii",
    "Dracaena trifasciata": "Sansevieria trifasciata",
    "Dypsis lutescens": "Chrysalidocarpus lutescens",
}

_API_LOCK = threading.Lock()
_LAST_API_TS = 0.0
_THREAD_LOCAL = threading.local()


def _get_http_session() -> requests.Session:
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        sess.headers.update({"User-Agent": INAT_USER_AGENT})
        _THREAD_LOCAL.session = sess
    return sess


def clean_binomial(raw_name: str) -> str:
    """Normalize an iNaturalist taxon name (including hybrids with '×') to a 2-word binomial."""
    tokens = [
        tok
        for tok in str(raw_name).replace("×", " ").split()
        if tok.lower() not in ("x", "×")
    ]
    if len(tokens) >= 2:
        return f"{tokens[0]} {tokens[1]}"
    return str(raw_name).strip()


def inat_api_get(
    endpoint: str,
    params: dict[str, object],
    cache_dir: Path | None = None,
) -> dict:
    """Thread-safe rate-limited GET against api.inaturalist.org (max 1 req/s), cached under cache_dir."""
    global _LAST_API_TS
    if cache_dir is None:
        cache_dir = CACHE_DIR / "inat_api"

    query = urllib.parse.urlencode(sorted((k, str(v)) for k, v in params.items()))
    url = f"{INAT_API}/{endpoint.lstrip('/')}?{query}"
    cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    cache_file = Path(cache_dir) / f"{cache_key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    sess = _get_http_session()
    for attempt in range(4):
        with _API_LOCK:
            now = time.monotonic()
            wait = 1.02 - (now - _LAST_API_TS)
            if wait > 0:
                time.sleep(wait)
            _LAST_API_TS = time.monotonic()

        try:
            resp = sess.get(url, headers={"Accept": "application/json"}, timeout=25)
            if resp.status_code == 429:
                time.sleep(3.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            try:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(data), encoding="utf-8")
            except Exception:
                pass
            return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))
    return {}


def s3_medium_url(photo: dict) -> tuple[str, str, str] | None:
    """Return (photo_id, s3_url, license_code) for a photo dict if its license is allowed."""
    if not isinstance(photo, dict):
        return None
    lic = str(photo.get("license_code") or "").strip().lower()
    if lic not in ALLOWED_LICENSES:
        return None
    pid = photo.get("id")
    if pid is None:
        return None
    photo_id = str(pid).strip()
    if not photo_id:
        return None
    raw_url = str(photo.get("url") or photo.get("medium_url") or "").strip()
    ext = "jpeg"
    if raw_url:
        m = re.search(r"\.([a-zA-Z0-9]+)(?:\?|$)", raw_url)
        if m:
            ext_cand = m.group(1).lower()
            if ext_cand in ("jpg", "jpeg", "png", "gif", "webp"):
                ext = ext_cand
    url = f"{S3_BASE}/{photo_id}/medium.{ext}"
    return photo_id, url, lic


def select_vl600_species(
    api_get: Callable[[str, dict[str, object]], dict] = inat_api_get,
    n_india: int = 400,
    n_world: int = 200,
) -> tuple[list[dict[str, object]], int]:
    """Select vl600 species by observation count ONLY:
    (a) ~400 most observed in India
    (b) ~200 most observed worldwide not in (a)
    (c) common-Indian list resolved to accepted iNat binomials.
    Returns (species_list, total_india_plant_observations).
    """
    india_entries: list[dict] = []
    page = 1
    while True:
        data = api_get(
            "observations/species_counts",
            {
                "place_id": INDIA_PLACE_ID,
                "taxon_id": PLANTAE_TAXON_ID,
                "quality_grade": "research",
                "hrank": "species",
                "lrank": "species",
                "per_page": 500,
                "page": page,
            },
        )
        results = data.get("results") or []
        india_entries.extend(results)
        total_res = int(data.get("total_results") or len(india_entries))
        if not results or len(india_entries) >= total_res or page >= 16:
            break
        page += 1

    total_india_obs = sum(int(r.get("count") or 0) for r in india_entries)

    india_by_name: dict[str, dict[str, object]] = {}
    india_by_tid: dict[int, int] = {}
    for r in india_entries:
        taxon = r.get("taxon") or {}
        if "id" not in taxon or "name" not in taxon:
            continue
        tid = int(taxon["id"])
        cnt = int(r.get("count") or 0)
        cn = clean_binomial(str(taxon["name"]))
        if len(cn.split()) != 2:
            continue
        india_by_tid[tid] = cnt
        if cn not in india_by_name:
            india_by_name[cn] = {
                "name": cn,
                "taxon_id": tid,
                "india_count": cnt,
                "world_count": int(taxon.get("observations_count") or cnt),
                "common_name": str(taxon.get("preferred_common_name") or "").strip(),
            }

    world_entries: list[dict] = []
    for wpage in (1, 2):
        wdata = api_get(
            "observations/species_counts",
            {
                "taxon_id": PLANTAE_TAXON_ID,
                "quality_grade": "research",
                "hrank": "species",
                "lrank": "species",
                "per_page": 500,
                "page": wpage,
            },
        )
        wres = wdata.get("results") or []
        world_entries.extend(wres)
        if len(wres) < 500:
            break

    world_by_name: dict[str, dict[str, object]] = {}
    for r in world_entries:
        taxon = r.get("taxon") or {}
        if "id" not in taxon or "name" not in taxon:
            continue
        tid = int(taxon["id"])
        wcnt = int(r.get("count") or 0)
        cn = clean_binomial(str(taxon["name"]))
        if len(cn.split()) != 2:
            continue
        if cn not in world_by_name:
            world_by_name[cn] = {
                "name": cn,
                "taxon_id": tid,
                "india_count": india_by_tid.get(tid, int(india_by_name.get(cn, {}).get("india_count", 0))),
                "world_count": wcnt,
                "common_name": str(taxon.get("preferred_common_name") or "").strip(),
            }

    # Resolve list (c) common Indian plants
    common_resolved: dict[str, dict[str, object]] = {}
    for common_label, target, is_genus in COMMON_INDIAN_SPEC:
        if is_genus:
            prefix = target + " "
            matches = [v for k, v in india_by_name.items() if k.startswith(prefix)]
            if matches:
                best = max(matches, key=lambda x: int(x["india_count"]))
                sp_name = str(best["name"])
                info = dict(best)
            else:
                tdata = api_get("taxa", {"q": target, "rank": "species", "per_page": 5})
                tres = tdata.get("results") or []
                if not tres:
                    continue
                t0 = tres[0]
                sp_name = clean_binomial(str(t0["name"]))
                tid = int(t0["id"])
                info = {
                    "name": sp_name,
                    "taxon_id": tid,
                    "india_count": india_by_tid.get(tid, 0),
                    "world_count": int(t0.get("observations_count") or 0),
                    "common_name": common_label,
                }
        else:
            accepted = INAT_SYNONYM_MAP.get(target, target)
            if accepted in india_by_name:
                info = dict(india_by_name[accepted])
                sp_name = accepted
            elif target in india_by_name:
                info = dict(india_by_name[target])
                sp_name = target
            elif accepted in world_by_name:
                info = dict(world_by_name[accepted])
                sp_name = accepted
            else:
                tdata = api_get("taxa", {"q": accepted, "per_page": 5})
                tres = tdata.get("results") or []
                if not tres:
                    continue
                t0 = tres[0]
                sp_name = clean_binomial(str(t0["name"]))
                tid = int(t0["id"])
                info = {
                    "name": sp_name,
                    "taxon_id": tid,
                    "india_count": india_by_tid.get(tid, 0),
                    "world_count": int(t0.get("observations_count") or 0),
                    "common_name": str(t0.get("preferred_common_name") or common_label),
                }
        if not info.get("common_name"):
            info["common_name"] = common_label
        common_resolved[sp_name] = info

    chosen: list[dict[str, object]] = []
    seen_names: set[str] = set()

    # (a) Top ~400 in India
    for r in india_entries:
        taxon = r.get("taxon") or {}
        if "id" not in taxon or "name" not in taxon:
            continue
        cn = clean_binomial(str(taxon["name"]))
        if len(cn.split()) != 2 or cn in seen_names:
            continue
        seen_names.add(cn)
        tid = int(taxon["id"])
        icnt = int(r.get("count") or 0)
        wcnt = int(world_by_name.get(cn, {}).get("world_count") or taxon.get("observations_count") or icnt)
        cname = str(taxon.get("preferred_common_name") or common_resolved.get(cn, {}).get("common_name") or "").strip()
        chosen.append(
            {
                "name": cn,
                "common_name": cname,
                "source": "india",
                "india_count": icnt,
                "world_count": wcnt,
                "count": icnt,
                "taxon_id": tid,
                "in_common_list": cn in common_resolved,
            }
        )
        if len(chosen) >= n_india:
            break

    # (b) Top ~200 worldwide not already in (a)
    n_world_added = 0
    for r in world_entries:
        taxon = r.get("taxon") or {}
        if "id" not in taxon or "name" not in taxon:
            continue
        cn = clean_binomial(str(taxon["name"]))
        if len(cn.split()) != 2 or cn in seen_names:
            continue
        seen_names.add(cn)
        tid = int(taxon["id"])
        wcnt = int(r.get("count") or 0)
        icnt = int(india_by_tid.get(tid, india_by_name.get(cn, {}).get("india_count", 0)))
        cname = str(taxon.get("preferred_common_name") or common_resolved.get(cn, {}).get("common_name") or "").strip()
        chosen.append(
            {
                "name": cn,
                "common_name": cname,
                "source": "world",
                "india_count": icnt,
                "world_count": wcnt,
                "count": wcnt,
                "taxon_id": tid,
                "in_common_list": cn in common_resolved,
            }
        )
        n_world_added += 1
        if n_world_added >= n_world:
            break

    # (c) Always add any remaining from the common-Indian list
    for cn, info in common_resolved.items():
        if cn in seen_names:
            continue
        seen_names.add(cn)
        icnt = int(info.get("india_count") or 0)
        wcnt = int(info.get("world_count") or icnt)
        chosen.append(
            {
                "name": cn,
                "common_name": str(info.get("common_name") or ""),
                "source": "common",
                "india_count": icnt,
                "world_count": wcnt,
                "count": icnt if icnt > 0 else wcnt,
                "taxon_id": int(info["taxon_id"]),
                "in_common_list": True,
            }
        )

    return chosen, total_india_obs


def write_species_csv(species_list: list[dict[str, object]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "common_name",
        "source",
        "count",
        "india_count",
        "world_count",
        "taxon_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sp in species_list:
            writer.writerow({k: sp.get(k, "") for k in fieldnames})


def _extract_obs_photos(
    obs: dict,
    target_taxon_id: int,
    allow_non_rg: bool,
    forced_in_india: bool | None = None,
) -> tuple[str, str, list[dict[str, object]]] | None:
    """Return (obs_id, period_key, photo_records) if observation is valid, else None.
    period_key is 'pre' (< 2025-07-01) or 'post' (>= 2025-07-01)."""
    if not isinstance(obs, dict) or "id" not in obs:
        return None
    obs_id = str(obs["id"]).strip()
    observed_on = str(
        obs.get("observed_on")
        or (obs.get("observed_on_details") or {}).get("date")
        or ""
    ).strip()[:10]
    if len(observed_on) != 10 or not re.match(r"^\d{4}-\d{2}-\d{2}$", observed_on):
        return None

    qg = str(obs.get("quality_grade") or "").strip().lower()
    if qg != "research":
        if not allow_non_rg:
            return None
        comm_tid = obs.get("community_taxon_id")
        taxon_obj = obs.get("taxon") or {}
        anc = taxon_obj.get("ancestor_ids") or []
        if comm_tid != target_taxon_id and not (
            comm_tid is not None
            and taxon_obj.get("id") == comm_tid
            and target_taxon_id in anc
        ):
            return None
        agreements = int(obs.get("num_identification_agreements") or 0)
        if agreements < 1:
            return None

    lat: float | str = ""
    lon: float | str = ""
    geojson = obs.get("geojson")
    if isinstance(geojson, dict):
        coords = geojson.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            try:
                lon_v, lat_v = float(coords[0]), float(coords[1])
                if np.isfinite(lat_v) and np.isfinite(lon_v):
                    lat, lon = lat_v, lon_v
            except Exception:
                pass
    if lat == "" and isinstance(obs.get("location"), str) and "," in str(obs["location"]):
        try:
            lat_s, lon_s = str(obs["location"]).split(",", 1)
            lat_v, lon_v = float(lat_s.strip()), float(lon_s.strip())
            if np.isfinite(lat_v) and np.isfinite(lon_v):
                lat, lon = lat_v, lon_v
        except Exception:
            pass

    place_ids = obs.get("place_ids") or []
    in_india = (
        bool(forced_in_india)
        if forced_in_india is not None
        else (INDIA_PLACE_ID in place_ids)
    )

    photo_recs: list[dict[str, object]] = []
    seen_pids: set[str] = set()
    for p in obs.get("photos") or []:
        info = s3_medium_url(p)
        if info is None:
            continue
        photo_id, s3_url, lic = info
        if photo_id in seen_pids:
            continue
        seen_pids.add(photo_id)
        photo_recs.append(
            {
                "obs_id": obs_id,
                "photo_id": photo_id,
                "lat": lat,
                "lon": lon,
                "observed_on": observed_on,
                "license": lic,
                "url": s3_url,
                "quality_grade": qg or "research",
                "in_india": in_india,
            }
        )
        if len(photo_recs) >= 3:
            break

    if not photo_recs:
        return None

    period = "post" if observed_on >= "2025-07-01" else "pre"
    return obs_id, period, photo_recs


def _choose_smart_d2(expected_count: int) -> str | None:
    """Choose a d2 cutoff date after 2025-07-01 so a single per_page=200 desc query
    captures ~25 post-cutoff observations and ~150 pre-cutoff observations."""
    if expected_count <= 450:
        return None
    days = max(1, min(480, int(round(30.0 / max(0.05, expected_count * 0.65 / 2000.0)))))
    target = date(2025, 7, 1) + timedelta(days=days)
    if target >= date(2026, 11, 15):
        return None
    return target.isoformat()


def fetch_photos_for_species(
    sp_info: dict[str, object],
    api_get: Callable[[str, dict[str, object]], dict] = inat_api_get,
    max_pre_photos: int = 80,
    max_post_obs: int = 15,
    prefetched_india_obs: list[dict] | None = None,
) -> list[dict[str, object]]:
    """Fetch up to `max_pre_photos` photos (< 2025-07-01) and up to `max_post_obs` observations (>= 2025-07-01),
    at most 3 photos/obs, preferring India then filling from worldwide. List (c) species may also use
    casual/needs_id observations with community_taxon_id == taxon_id and >= 1 agreeing ID."""
    taxon_id = int(sp_info["taxon_id"])
    india_count = int(sp_info.get("india_count") or 0)
    world_count = int(sp_info.get("world_count") or india_count)
    in_common = bool(sp_info.get("in_common_list", False) or sp_info.get("source") == "common")

    pre_photos: list[dict[str, object]] = []
    post_obs_count = 0
    post_photos: list[dict[str, object]] = []
    seen_obs: set[str] = set()

    def _ingest(obs_list: list[dict], allow_non_rg: bool, forced_india: bool | None) -> None:
        nonlocal post_obs_count
        for obs in obs_list:
            if len(pre_photos) >= max_pre_photos and post_obs_count >= max_post_obs:
                break
            parsed = _extract_obs_photos(obs, taxon_id, allow_non_rg, forced_india)
            if parsed is None:
                continue
            obs_id, period, recs = parsed
            if obs_id in seen_obs:
                continue
            if period == "post":
                if post_obs_count >= max_post_obs:
                    continue
                seen_obs.add(obs_id)
                post_obs_count += 1
                post_photos.extend(recs[:3])
            else:
                if len(pre_photos) >= max_pre_photos:
                    continue
                seen_obs.add(obs_id)
                remaining = max_pre_photos - len(pre_photos)
                pre_photos.extend(recs[: min(3, remaining)])

    def _query_scope(place_id: int | None, expected_cnt: int, quality_grade: str | None) -> None:
        if len(pre_photos) >= max_pre_photos and post_obs_count >= max_post_obs:
            return
        base_params: dict[str, object] = {
            "taxon_id": taxon_id,
            "photos": "true",
            "photo_license": "cc0,cc-by,cc-by-nc",
            "order_by": "observed_on",
            "order": "desc",
        }
        if place_id is not None:
            base_params["place_id"] = place_id
        if quality_grade is not None:
            base_params["quality_grade"] = quality_grade
        allow_non_rg = quality_grade is None
        forced_india = True if place_id == INDIA_PLACE_ID else None

        # Try a single combined call first
        d2_smart: str | None = None
        res1_len = 0
        tried_combined = False
        if len(pre_photos) < max_pre_photos and post_obs_count < max_post_obs:
            tried_combined = True
            p1 = dict(base_params)
            d2_smart = _choose_smart_d2(expected_cnt)
            p1["per_page"] = 140 if d2_smart is not None else 200
            if d2_smart is not None:
                p1["d2"] = d2_smart
            res1 = api_get("observations", p1).get("results") or []
            res1_len = len(res1)
            _ingest(res1, allow_non_rg, forced_india)

        # Follow-up for post-cutoff only if we didn't already query from today downwards
        if post_obs_count < max_post_obs and expected_cnt > 0 and (not tried_combined or d2_smart is not None):
            p_post = dict(base_params)
            p_post["d1"] = "2025-07-01"
            p_post["per_page"] = 25
            res_post = api_get("observations", p_post).get("results") or []
            _ingest(res_post, allow_non_rg, forced_india)

        # Follow-up for pre-cutoff only if combined call wasn't tried or hit page limit before getting 80 photos
        page_cap = 140 if d2_smart is not None else 200
        if len(pre_photos) < max_pre_photos and expected_cnt > 0 and (not tried_combined or res1_len >= page_cap):
            p_pre = dict(base_params)
            p_pre["d2"] = "2025-06-30"
            p_pre["per_page"] = 65
            res_pre = api_get("observations", p_pre).get("results") or []
            _ingest(res_pre, allow_non_rg, forced_india)

    # 1. India research-grade
    if prefetched_india_obs is not None:
        _ingest(prefetched_india_obs, allow_non_rg=False, forced_india=True)
    elif india_count > 0:
        _query_scope(INDIA_PLACE_ID, india_count, "research")

    # 2. Worldwide research-grade fill
    if (len(pre_photos) < max_pre_photos or post_obs_count < max_post_obs) and world_count > india_count:
        _query_scope(None, world_count, "research")

    # 3. For list (c) common-Indian species, allow casual/needs_id if still short
    if in_common and (len(pre_photos) < max_pre_photos or post_obs_count < max_post_obs):
        _query_scope(INDIA_PLACE_ID, max(india_count, 100), None)
        if len(pre_photos) < max_pre_photos or post_obs_count < max_post_obs:
            _query_scope(None, max(world_count, 200), None)

    return pre_photos + post_photos


def download_s3_photo(task: tuple[Path, str]) -> bool:
    """Download a medium photo from iNaturalist open-data S3 with retries and backoff."""
    dest, url = task
    if dest.exists() and dest.stat().st_size > 100:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = _get_http_session()

    alt_urls = [url]
    if url.endswith(".jpeg"):
        alt_urls.append(url[:-5] + ".jpg")
    elif url.endswith(".jpg"):
        alt_urls.append(url[:-4] + ".jpeg")

    for attempt in range(3):
        for candidate_url in alt_urls:
            try:
                resp = sess.get(candidate_url, timeout=20)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                raw = resp.content
                if len(raw) <= 100:
                    continue
                if raw[:2] == b"\xff\xd8":
                    dest.write_bytes(raw)
                else:
                    with Image.open(BytesIO(raw)) as im:
                        im.convert("RGB").save(dest, format="JPEG", quality=90)
                return True
            except Exception:
                pass
        if attempt < 2:
            time.sleep(0.8 * (attempt + 1))
    return False


def check_disk_budget(
    planned_photos: int,
    free_bytes: int,
    log_fn: Callable[[str], None] = print,
) -> tuple[bool, float, float]:
    """Log planned photo count and expected GB; return (ok, expected_gb, free_gb)."""
    expected_gb = (planned_photos * EXPECTED_BYTES_PER_PHOTO) / 1e9
    free_gb = free_bytes / 1e9
    remaining_gb = free_gb - expected_gb
    log_fn(
        f"Planned photo count: {planned_photos}, expected GB: {expected_gb:.3f} GB "
        f"(free disk: {free_gb:.2f} GB, free after: {remaining_gb:.2f} GB)"
    )
    if expected_gb > MAX_EXPECTED_GB or remaining_gb < MIN_FREE_GB_AFTER:
        log_fn(
            f"BLOCKED: planned_photos={planned_photos} expected_gb={expected_gb:.3f} "
            f"free_gb={free_gb:.2f} remaining_gb={remaining_gb:.2f}"
        )
        return False, expected_gb, free_gb
    return True, expected_gb, free_gb


def _batch_prefetch_small_india(
    species_list: list[dict[str, object]],
    checkpoint_dir: Path,
    api_get: Callable[[str, dict[str, object]], dict],
) -> dict[int, list[dict]]:
    """Batch-fetch India observations for world species with 0 < india_count <= 180 to save API calls."""
    small = [
        sp
        for sp in species_list
        if sp.get("source") == "world"
        and 0 < int(sp.get("india_count") or 0) <= 180
        and not (checkpoint_dir / f"{sp['taxon_id']}.json").exists()
    ]
    if not small:
        return {}

    bins: list[list[int]] = []
    cur_bin: list[int] = []
    cur_sum = 0
    for sp in small:
        cnt = int(sp["india_count"])
        if cur_bin and (cur_sum + cnt > 180 or len(cur_bin) >= 15):
            bins.append(cur_bin)
            cur_bin = []
            cur_sum = 0
        cur_bin.append(int(sp["taxon_id"]))
        cur_sum += cnt
    if cur_bin:
        bins.append(cur_bin)

    out: dict[int, list[dict]] = {int(sp["taxon_id"]): [] for sp in small}
    target_set = set(out.keys())
    for b in bins:
        data = api_get(
            "observations",
            {
                "place_id": INDIA_PLACE_ID,
                "taxon_id": ",".join(str(t) for t in b),
                "quality_grade": "research",
                "photos": "true",
                "photo_license": "cc0,cc-by,cc-by-nc",
                "per_page": 200,
                "order_by": "observed_on",
                "order": "desc",
            },
        )
        for obs in data.get("results") or []:
            taxon = obs.get("taxon") or {}
            tid = taxon.get("id")
            if isinstance(tid, int) and tid in target_set:
                out[tid].append(obs)
            else:
                for anc in taxon.get("ancestor_ids") or []:
                    if anc in target_set:
                        out[anc].append(obs)
                        break
    return out


def build_vl600(
    root: Path = DATA_DIR / "vl600",
    cache_dir: Path = CACHE_DIR,
    runs_dir: Path = RUNS_DIR,
    api_get: Callable[[str, dict[str, object]], dict] = inat_api_get,
    download_fn: Callable[[tuple[Path, str]], bool] = download_s3_photo,
    free_bytes_fn: Callable[[Path], int] | None = None,
    n_india: int = 400,
    n_world: int = 200,
) -> dict[str, object]:
    """Build vl600 dataset, manifests, and runs/L1/vl600_stats.json."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir)
    runs_dir = Path(runs_dir)

    log_path = runs_dir / "logs" / "build_vl600.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()

    def _log(msg: str) -> None:
        print(msg, flush=True)
        with log_lock:
            try:
                with log_path.open("a", encoding="utf-8") as lf:
                    lf.write(msg + "\n")
            except Exception:
                pass

    _log("[vl600] Step 1: selecting species by observation count...")
    species_list, total_india_obs = select_vl600_species(
        api_get=api_get, n_india=n_india, n_world=n_world
    )
    write_species_csv(species_list, root / "species.csv")

    covered_india_obs = sum(int(sp.get("india_count") or 0) for sp in species_list)
    coverage_pct = (
        round((covered_india_obs / total_india_obs) * 100.0, 2)
        if total_india_obs > 0
        else 0.0
    )
    _log(
        f"[vl600] Selected {len(species_list)} species "
        f"(india={sum(1 for s in species_list if s['source'] == 'india')}, "
        f"world={sum(1 for s in species_list if s['source'] == 'world')}, "
        f"common={sum(1 for s in species_list if s['source'] == 'common')}); "
        f"India RG coverage={coverage_pct:.2f}% ({covered_india_obs}/{total_india_obs})"
    )

    checkpoint_dir = cache_dir / "vl600" / "species_obs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    prefetched_india = _batch_prefetch_small_india(species_list, checkpoint_dir, api_get)

    species_photos: dict[str, list[dict[str, object]]] = {}
    todo_species: list[dict[str, object]] = []
    for sp in species_list:
        tid = int(sp["taxon_id"])
        cp_file = checkpoint_dir / f"{tid}.json"
        if cp_file.exists():
            try:
                species_photos[str(sp["name"])] = json.loads(cp_file.read_text(encoding="utf-8"))
                continue
            except Exception:
                pass
        todo_species.append(sp)

    _log(
        f"[vl600] Step 2a: fetching observation metadata for {len(todo_species)} species "
        f"({len(species_photos)} already checkpointed)..."
    )

    if todo_species:
        done_count = 0
        lock = threading.Lock()

        def _fetch_one_species(sp: dict[str, object]) -> tuple[str, list[dict[str, object]]]:
            nonlocal done_count
            tid = int(sp["taxon_id"])
            sp_name = str(sp["name"])
            recs = fetch_photos_for_species(
                sp,
                api_get=api_get,
                prefetched_india_obs=prefetched_india.get(tid),
            )
            cp_file = checkpoint_dir / f"{tid}.json"
            try:
                cp_file.write_text(json.dumps(recs), encoding="utf-8")
            except Exception:
                pass
            with lock:
                done_count += 1
                if done_count % 25 == 0 or done_count == len(todo_species):
                    _log(f"[vl600] API metadata progress: {done_count}/{len(todo_species)} species")
            return sp_name, recs

        with ThreadPoolExecutor(max_workers=4) as pool:
            for sp_name, recs in pool.map(_fetch_one_species, todo_species):
                species_photos[sp_name] = recs

    download_tasks: list[tuple[Path, str]] = []
    all_records: list[tuple[str, dict[str, object], Path]] = []
    for sp in sorted(species_list, key=lambda x: str(x["name"])):
        sp_name = str(sp["name"])
        recs = species_photos.get(sp_name) or []
        for r in recs:
            fname = f"{r['obs_id']}_{r['photo_id']}.jpg"
            dest = root / sp_name / fname
            download_tasks.append((dest, str(r["url"])))
            all_records.append((sp_name, r, dest))

    if free_bytes_fn is not None:
        free_bytes = free_bytes_fn(root)
    else:
        free_bytes = shutil.disk_usage(root).free

    ok_budget, expected_gb, free_gb = check_disk_budget(
        len(download_tasks), free_bytes, log_fn=_log
    )
    if not ok_budget:
        return {
            "status": "BLOCKED",
            "planned_photos": len(download_tasks),
            "expected_gb": round(expected_gb, 4),
            "free_gb": round(free_gb, 4),
        }

    existing_count = sum(1 for dest, _ in download_tasks if dest.exists() and dest.stat().st_size > 100)
    _log(
        f"[vl600] Step 2b: downloading {len(download_tasks)} photos with 8 workers "
        f"({existing_count} already on disk)..."
    )

    dl_done = 0
    dl_ok = 0
    dl_lock = threading.Lock()
    total_dl = len(download_tasks)

    def _tracked_download(task: tuple[Path, str]) -> bool:
        nonlocal dl_done, dl_ok
        ok = download_fn(task)
        with dl_lock:
            dl_done += 1
            if ok:
                dl_ok += 1
            if dl_done % 2000 == 0 or dl_done == total_dl:
                _log(f"[vl600] Download progress: {dl_done}/{total_dl} (ok={dl_ok})")
        return ok

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_tracked_download, download_tasks))

    ok_paths = {task[0] for task, ok in zip(download_tasks, results) if ok}

    meta_rows: list[dict[str, object]] = []
    total_bytes = 0
    for sp_name, r, dest in all_records:
        if dest not in ok_paths:
            continue
        try:
            total_bytes += dest.stat().st_size
        except Exception:
            pass
        meta_rows.append(
            {
                "image": f"{sp_name}/{dest.name}",
                "obs_id": r["obs_id"],
                "photo_id": r["photo_id"],
                "lat": r["lat"],
                "lon": r["lon"],
                "observed_on": r["observed_on"],
                "license": r["license"],
                "url": r["url"],
                "quality_grade": r["quality_grade"],
                "in_india": r["in_india"],
            }
        )

    meta_path = root / "meta.csv"
    meta_fields = [
        "image",
        "obs_id",
        "photo_id",
        "lat",
        "lon",
        "observed_on",
        "license",
        "url",
        "quality_grade",
        "in_india",
    ]
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=meta_fields)
        writer.writeheader()
        writer.writerows(meta_rows)

    manifests_dir = cache_dir / "manifests"
    split_info = _process_vl600(root, manifests_dir)

    campus_dir = root.parent / "campus"
    campus_info: dict[str, object] | None = None
    if campus_dir.exists():
        campus_info = _process_campus(
            campus_dir, manifests_dir, vl600_taxa=split_info["taxa"]  # type: ignore[arg-type]
        )
    else:
        _log("[vl600] $VANALENS_DATA/campus does not exist; skipping campus_test manifest.")

    actual_gb = round(total_bytes / 1e9, 4)
    stats: dict[str, object] = {
        "species": len(species_list),
        "species_by_source": {
            "india": sum(1 for s in species_list if s["source"] == "india"),
            "world": sum(1 for s in species_list if s["source"] == "world"),
            "common": sum(1 for s in species_list if s["source"] == "common"),
        },
        "photos": len(meta_rows),
        "photos_per_split": split_info["photos_per_split"],
        "observations_per_split": split_info["observations_per_split"],
        "gb": actual_gb,
        "coverage_pct": coverage_pct,
        "covered_india_observations": covered_india_obs,
        "total_india_observations": total_india_obs,
        "species_lt5_test_obs": split_info["species_lt5_test_obs"],
        "species_with_lt5_test_observations": split_info["species_lt5_list"],
        "campus": campus_info if campus_info is not None else "skipped (no $VANALENS_DATA/campus)",
    }

    stats_path = runs_dir / "L1" / "vl600_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _log(
        f"[vl600] DONE: species={stats['species']} photos={stats['photos']} "
        f"gb={actual_gb} coverage_pct={coverage_pct}% "
        f"lt5_test_obs={stats['species_lt5_test_obs']}"
    )
    return stats


def main() -> None:
    build_vl600()


if __name__ == "__main__":
    main()
