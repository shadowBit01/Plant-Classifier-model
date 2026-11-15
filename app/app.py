"""Lane L5 (app). Gradio demo. Develop with VANALENS_MOCK=1; the integration lane switches it to the real model.

Run: VANALENS_MOCK=1 python -m app.app

Requirements:
 1 Accept 1 to 3 photos of the same plant and send them as one observation to the pipeline
   (pass the list to VanaLens.predict; if the pipeline only takes one photo, average the calibrated
   log-probabilities in an app helper).
 2 When not confident (conformal set has more than one species, or top probability below the state threshold),
   show "Not sure: A, B, C" with the set members and their probabilities.
 3 When the open-set reject fires, show "Not in catalog, closest genus/family X" with an "Add this species" option
   (name + up to 10 photos, calls add_species, no retraining).
 4 Evidence panel: nearest reference photos (Prediction.evidence) + matched traits.
 5 If Prediction.reranked: show the rationale in a collapsible box.
 6 Hint row: if the user sent one non-flower photo and the set has > 1 species, suggest a flower photo.
 7 "Add a species" tab: name + optional genus/family + up to 10 photos -> pipeline.add_species(...).
 8 Works on a phone browser (single column below 700 px). No external CSS/CDN needed.
Never call models directly: only vanalens.pipeline.get_pipeline().
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import urllib.parse

import gradio as gr
import numpy as np
import requests

from vanalens import config as vanalens_config
from vanalens import hierarchy
from vanalens.config import ALPHA, BACKOFF_TAU
from vanalens.contracts import ORGANS, Prediction, Taxon
from vanalens.pipeline import get_pipeline

os.environ.setdefault("VANALENS_MOCK", "0")

_PIPELINE = None
_COMMON_NAMES_CACHE: dict[str, str | None] = {}


def _load_csv_common_names(csv_path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    if not csv_path.is_file():
        return names
    try:
        with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                field_map = {fn.strip().lower(): fn for fn in reader.fieldnames if fn}
                sp_col = None
                for candidate in ("species", "name", "scientific_name", "scientificname", "scientific name", "taxon"):
                    if candidate in field_map:
                        sp_col = field_map[candidate]
                        break
                cn_col = None
                for candidate in (
                    "common_name",
                    "preferred_common_name",
                    "vernacular_name",
                    "english_common_name",
                    "commonname",
                    "common_names",
                    "common name",
                ):
                    if candidate in field_map:
                        cn_col = field_map[candidate]
                        break
                if sp_col and cn_col:
                    for row in reader:
                        sp = str(row.get(sp_col) or "").strip()
                        cn = str(row.get(cn_col) or "").strip()
                        if sp and cn:
                            names[sp] = cn
    except Exception:
        pass
    return names


def _save_common_name_to_cache(species: str, common_name: str) -> None:
    try:
        cache_file = vanalens_config.CACHE_DIR / "common_names.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if cache_file.is_file():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data[species] = common_name
        cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def load_common_names(dataset: str | None = None) -> dict[str, str]:
    """Read local DATA_DIR/<dataset>/species.csv, CACHE_DIR/<dataset>/species.csv, and CACHE_DIR/common_names.json."""
    loaded: dict[str, str] = {}
    ds = dataset or os.environ.get("VANALENS_DATASET", "inat_india")

    # 1. DATA_DIR/<dataset>/species.csv and DATA_DIR/species.csv
    if ds:
        p_data_ds = vanalens_config.DATA_DIR / ds / "species.csv"
        loaded.update(_load_csv_common_names(p_data_ds))
    p_data = vanalens_config.DATA_DIR / "species.csv"
    if p_data.is_file():
        loaded.update(_load_csv_common_names(p_data))

    # 2. CACHE_DIR/<dataset>/species.csv and CACHE_DIR/species.csv
    if ds:
        p_cache_ds = vanalens_config.CACHE_DIR / ds / "species.csv"
        loaded.update(_load_csv_common_names(p_cache_ds))
    p_cache = vanalens_config.CACHE_DIR / "species.csv"
    if p_cache.is_file():
        loaded.update(_load_csv_common_names(p_cache))

    # 3. CACHE_DIR/common_names.json
    cache_json = vanalens_config.CACHE_DIR / "common_names.json"
    if cache_json.is_file():
        try:
            data = json.loads(cache_json.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if k and v:
                        loaded[str(k).strip()] = str(v).strip()
        except Exception:
            pass

    _COMMON_NAMES_CACHE.update(loaded)
    return loaded


def get_common_name(
    species: str,
    gbif_key: int | None = None,
    fetch_api: bool | None = None,
) -> str | None:
    """Resolve common name for a species using local caches, falling back to iNaturalist / GBIF APIs."""
    if not species or not str(species).strip():
        return None
    species = str(species).strip()

    # 1. In-memory cache
    if species in _COMMON_NAMES_CACHE and _COMMON_NAMES_CACHE[species]:
        return _COMMON_NAMES_CACHE[species]

    # 2. Check CACHE_DIR/common_names.json directly if not in memory
    cache_file = vanalens_config.CACHE_DIR / "common_names.json"
    if cache_file.is_file():
        try:
            cdata = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cdata, dict) and species in cdata and cdata[species]:
                cname = str(cdata[species]).strip()
                _COMMON_NAMES_CACHE[species] = cname
                return cname
        except Exception:
            pass

    # 3. If missing and fetch_api is enabled (default True when VANALENS_MOCK != '1')
    if fetch_api is None:
        fetch_api = (os.environ.get("VANALENS_MOCK", "0") != "1")

    if not fetch_api:
        return _COMMON_NAMES_CACHE.get(species)

    cname = None
    # 4. Query iNaturalist API v1
    try:
        url = "https://api.inaturalist.org/v1/taxa"
        params = {
            "q": species,
            "rank": "species",
            "per_page": 1,
            "preferred_place_id": 6681,
        }
        resp = requests.get(
            url,
            params=params,
            timeout=3.0,
            headers={"User-Agent": "VanaLens/1.0"},
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                t = results[0]
                val = t.get("preferred_common_name") or t.get("english_common_name")
                if val and str(val).strip():
                    cname = str(val).strip()
    except Exception:
        pass

    # 5. GBIF fallback
    if not cname:
        try:
            resolved_key = gbif_key
            if resolved_key is None:
                match_url = "https://api.gbif.org/v1/species/match"
                resp = requests.get(
                    match_url,
                    params={"name": species},
                    timeout=3.0,
                    headers={"User-Agent": "VanaLens/1.0"},
                )
                if resp.status_code == 200:
                    mdata = resp.json()
                    resolved_key = mdata.get("usageKey") or mdata.get("speciesKey") or mdata.get("key")
            if resolved_key is not None:
                vn_url = f"https://api.gbif.org/v1/species/{resolved_key}/vernacularNames"
                resp = requests.get(
                    vn_url,
                    params={"limit": 20},
                    timeout=3.0,
                    headers={"User-Agent": "VanaLens/1.0"},
                )
                if resp.status_code == 200:
                    vdata = resp.json()
                    results = vdata.get("results", [])
                    eng_name = None
                    first_name = None
                    for r in results:
                        vname = r.get("vernacularName")
                        if vname and str(vname).strip():
                            if first_name is None:
                                first_name = str(vname).strip()
                            lang = str(r.get("language") or "").strip().lower()
                            if lang in ("eng", "en"):
                                eng_name = str(vname).strip()
                                break
                    cname = eng_name or first_name
        except Exception:
            pass

    if cname:
        _COMMON_NAMES_CACHE[species] = cname
        _save_common_name_to_cache(species, cname)
        return cname
    else:
        _COMMON_NAMES_CACHE[species] = None
        return None


def format_species_name(species: str, common_name: str | None = None) -> str:
    """Return 'Common Name (Scientific Name)' when common_name exists and differs from species."""
    if not species:
        return common_name or ""
    sp = str(species).strip()
    if common_name and str(common_name).strip():
        cn = str(common_name).strip()
        if cn.lower() != sp.lower():
            return f"{cn} ({sp})"
    return sp


def load_pipeline():
    """Sync cache/data paths from environment variables and load the pipeline via get_pipeline()."""
    os.environ.setdefault("VANALENS_MOCK", "0")
    dataset = os.environ.get("VANALENS_DATASET", "inat_india")
    state_env = os.environ.get("VANALENS_STATE")
    if state_env:
        sp = Path(state_env)
        if sp.suffix == ".json":
            dataset = sp.stem
            if sp.parent.name == "state":
                vanalens_config.CACHE_DIR = sp.parent.parent
        elif sp.is_dir():
            vanalens_config.CACHE_DIR = sp
    if "VANALENS_CACHE" in os.environ:
        vanalens_config.CACHE_DIR = Path(os.environ["VANALENS_CACHE"])
    if "VANALENS_DATA" in os.environ:
        vanalens_config.DATA_DIR = Path(os.environ["VANALENS_DATA"])
    elif (vanalens_config.CACHE_DIR / dataset / "labels.json").exists():
        vanalens_config.DATA_DIR = vanalens_config.CACHE_DIR
    load_common_names(dataset)
    if dataset != "inat_india":
        return get_pipeline(dataset)
    return get_pipeline()


# Try to import read_exif_gps; it may be NotImplementedError at runtime (dev stub).
try:
    from vanalens.data import read_exif_gps as _read_exif_gps
except Exception:
    _read_exif_gps = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_exif_gps(path: str) -> tuple[float, float] | None:
    """Return (lat, lon) from EXIF or None if unavailable / not implemented."""
    if _read_exif_gps is None:
        return None
    try:
        return _read_exif_gps(path)
    except Exception:
        return None


def _get_confidence_threshold(pipeline) -> float:
    """Determine confidence threshold from pipeline.state or default to BACKOFF_TAU."""
    state = getattr(pipeline, "state", None)
    if isinstance(state, dict):
        for k in ("confidence_threshold", "prob_threshold", "threshold", "tau"):
            if k in state and isinstance(state[k], (int, float)):
                return float(state[k])
        # If 'thr' is present in state and represents a probability threshold (0.0 < thr <= 1.0)
        if "thr" in state and isinstance(state["thr"], (int, float)) and 0.0 < state["thr"] <= 1.0:
            return float(state["thr"])
    if hasattr(pipeline, "tau") and isinstance(pipeline.tau, (int, float)):
        return float(pipeline.tau)
    return float(getattr(pipeline, "tau", BACKOFF_TAU))


def is_open_set_reject(pred: Prediction) -> bool:
    """Check if the open-set reject gate fired for this prediction."""
    if getattr(pred, "unknown", False) or getattr(pred, "is_unknown", False):
        return True
    if getattr(pred, "rank", "") == "unknown":
        rationale = getattr(pred, "rationale", "").lower()
        if "organ gate" in rationale or "not a plant" in rationale or "not-a-plant" in rationale:
            return False
        return True
    return False


def is_confident(pred: Prediction, pipeline) -> bool:
    """Check if the prediction is confident.

    Not confident if:
      - Conformal set has more than one species (len(pred.pred_set) > 1), OR
      - Top probability is below the state threshold.
    """
    if len(pred.pred_set) > 1:
        return False
    threshold = _get_confidence_threshold(pipeline)
    if getattr(pred, "top_k", None) and getattr(pred, "probs", None) is not None and len(pred.probs):
        top_prob = float(pred.probs[pred.top_k[0][0]])
    else:
        top_prob = float(getattr(pred, "confidence", 0.0))
    if top_prob < threshold:
        return False
    return True


def _closest_taxon_info(pred: Prediction, pipeline) -> tuple[str, str]:
    """Return (closest_genus_or_family, family) for open-set reject."""
    taxa = getattr(pipeline, "taxa", [])
    top_idx = pred.top_k[0][0] if getattr(pred, "top_k", None) else 0
    if taxa and top_idx < len(taxa):
        t = taxa[top_idx]
        genus = getattr(t, "genus", "").strip()
        fam = getattr(t, "family", "").strip()
        closest = genus if genus else fam
        return (closest or "unknown", fam)
    return ("unknown", "")


def average_log_probabilities(preds: list[Prediction], taxa: list[Taxon]) -> Prediction:
    """App helper: average calibrated log-probabilities across multiple photos of one plant observation."""
    if not preds:
        raise ValueError("preds must not be empty")
    if len(preds) == 1:
        return preds[0]

    log_probs = [np.log(np.maximum(pr.probs, 1e-12)) for pr in preds]
    mean_logp = np.mean(log_probs, axis=0)
    p = np.exp(mean_logp - mean_logp.max())
    p /= p.sum()

    order = np.argsort(-p, kind="stable")
    top_k = [(int(i), float(p[i])) for i in order[:5]]

    combined_set = []
    for pr in preds:
        for idx in pr.pred_set:
            if int(idx) not in combined_set:
                combined_set.append(int(idx))
    if not combined_set:
        combined_set = [int(order[0])]

    rank, label, conf = hierarchy.backoff(p, taxa, BACKOFF_TAU)

    evidence = []
    for pr in preds:
        for ev in pr.evidence:
            if ev not in evidence:
                evidence.append(ev)
    evidence = evidence[:3]

    traits = []
    for pr in preds:
        for tr in pr.traits:
            if tr not in traits:
                traits.append(tr)

    organ = preds[0].organ
    reranked = any(pr.reranked for pr in preds)
    rationales = [pr.rationale for pr in preds if pr.rationale]
    rationale = " | ".join(rationales)

    return Prediction(
        probs=p,
        top_k=top_k,
        pred_set=combined_set,
        rank=rank,
        label=label,
        confidence=float(conf),
        evidence=evidence,
        traits=traits,
        organ=organ,
        reranked=reranked,
        rationale=rationale,
    )


def predict_observation(
    pipeline,
    paths: list[str],
    lat: float | None = None,
    lon: float | None = None,
    use_rerank: bool = True,
) -> Prediction:
    """Send 1 to 3 photos of the same plant observation to the pipeline.

    Passes the list to pipeline.predict. If the pipeline only takes one photo
    (e.g. raises TypeError or AttributeError on list), averages calibrated log-probabilities
    across photos using average_log_probabilities.
    """
    if not paths:
        raise ValueError("predict_observation requires at least one photo path")
    if len(paths) > 3:
        raise ValueError(f"predict_observation accepts 1 to 3 photos (got {len(paths)})")

    try:
        return pipeline.predict(list(paths), lat=lat, lon=lon, use_rerank=use_rerank)
    except (TypeError, AttributeError):
        preds = []
        for p in paths:
            try:
                pr = pipeline.predict(p, lat=lat, lon=lon, use_rerank=use_rerank)
            except TypeError:
                pr = pipeline.predict(p, lat=lat, lon=lon)
            preds.append(pr)
        taxa = getattr(pipeline, "taxa", [])
        return average_log_probabilities(preds, taxa)


def _chips_html(
    pred_set: list[int],
    taxa_map: dict[int, str],
    probs: np.ndarray | None = None,
    taxa_by_idx: dict[int, Taxon] | None = None,
) -> str:
    """Render the conformal prediction set as inline HTML chips with probabilities."""
    cov_pct = int(round((1 - ALPHA) * 100))
    chip_items = []
    for i in pred_set:
        name = taxa_map.get(i, f"class {i}")
        t = taxa_by_idx.get(i) if taxa_by_idx else None
        gbif_key = getattr(t, "gbif_key", None) if t else None
        cname = get_common_name(name, gbif_key=gbif_key)
        formatted_name = format_species_name(name, cname)
        if probs is not None and i < len(probs):
            prob_str = f" ({probs[i] * 100:.1f}%)"
        else:
            prob_str = ""
        chip_items.append(
            f'<span style="display:inline-block;background:#e8f5e9;border:1px solid #81c784;'
            f'border-radius:16px;padding:2px 10px;margin:2px 3px;font-size:0.85em;color:#1b5e20">'
            f'{formatted_name}{prob_str}</span>'
        )
    chips = "".join(chip_items)
    return (
        f'<div style="margin-top:6px">'
        f'<em style="color:#555">{cov_pct}% sure it\'s one of these:</em><br>{chips}'
        f'</div>'
    )


def _traits_html(pred: Prediction) -> str:
    if getattr(pred, "traits", None):
        items = "".join(f"<li>{t}</li>" for t in pred.traits)
        return f"<ul style='margin:4px 0 0 16px;padding:0;color:#1a1a1a'>{items}</ul>"
    return ""


def _evidence_html(pred: Prediction) -> str:
    if getattr(pred, "evidence", None):
        spans = "".join(
            f'<span style="display:inline-block;background:#f5f5f5;border:1px solid #ddd;'
            f'border-radius:4px;padding:2px 7px;margin:2px 2px;font-size:0.8em;font-family:monospace;color:#1a1a1a">'
            f'{e}</span>'
            for e in pred.evidence[:3]
        )
        return f'<div style="margin-top:8px;color:#1a1a1a"><strong>Nearest references:</strong><br>{spans}</div>'
    return ""


def _rationale_html(pred: Prediction) -> str:
    if getattr(pred, "reranked", False) and getattr(pred, "rationale", ""):
        return (
            f'<details style="margin-top:8px"><summary style="cursor:pointer;color:#1565c0">'
            f'🔍 Reranking rationale</summary>'
            f'<p style="margin:4px 8px;color:#333">{pred.rationale}</p></details>'
        )
    return ""


def _result_html(pred: Prediction, pipeline) -> str:
    """Build the full result card HTML from a Prediction."""
    taxa_map = {t.idx: t.species for t in pipeline.taxa} if hasattr(pipeline, "taxa") else {}
    taxa_by_idx = {t.idx: t for t in pipeline.taxa} if hasattr(pipeline, "taxa") else {}

    # Case 1: Organ gate reject
    organ = getattr(pred, "organ", None)
    rationale = getattr(pred, "rationale", "")
    if organ == "other" or "organ gate" in rationale.lower():
        return (
            '<div style="border:2px solid #e57373;border-radius:12px;padding:16px;'
            'background:#ffebee;margin:8px 0;color:#1a1a1a">'
            '<div style="font-size:1.3em;font-weight:700;color:#c62828">'
            'Photo does not look like a plant</div>'
            '<p style="color:#666;margin:4px 0 0">'
            f'{rationale or "The photo does not appear to contain a plant."}</p>'
            '</div>'
        )

    # Case 2: Open-set reject fires
    if is_open_set_reject(pred):
        closest, fam = _closest_taxon_info(pred, pipeline)
        fam_html = f'<div style="color:#666;font-size:0.9em;margin-top:2px">Family: {fam}</div>' if fam and fam != closest else ""
        evidence_html = _evidence_html(pred)
        rationale_html = _rationale_html(pred)
        return (
            '<div style="border:2px solid #ef5350;border-radius:12px;padding:16px;'
            'background:#fff8e1;margin:8px 0;color:#1a1a1a">'
            f'<div style="font-size:1.3em;font-weight:700;color:#c62828">'
            f'Not in catalog, closest genus/family {closest}</div>'
            f'{fam_html}'
            '<p style="color:#555;margin:6px 0 8px">'
            'This plant appears to be a species not currently in our catalog.'
            '</p>'
            '<div style="margin-top:12px;padding:12px;background:#e8f5e9;border:1px solid #81c784;border-radius:8px">'
            '<div style="font-weight:600;color:#2e7d32;font-size:1.05em">🌱 Add this species</div>'
            '<p style="margin:4px 0 0;font-size:0.9em;color:#333">'
            'Know this plant? Use the <strong>"Add this species"</strong> option below to add it to the catalog '
            '(name + up to 10 photos, calls add_species with no retraining).'
            '</p>'
            '</div>'
            f'{evidence_html}'
            f'{rationale_html}'
            '</div>'
        )

    # Case 3: Not confident (conformal set has > 1 species, or top probability below state threshold)
    if not is_confident(pred, pipeline):
        set_members = []
        for i in pred.pred_set:
            sp = taxa_map.get(i, f"Taxon {i}")
            t = taxa_by_idx.get(i)
            cname = get_common_name(sp, gbif_key=getattr(t, "gbif_key", None) if t else None)
            set_members.append(format_species_name(sp, cname))
        members_str = ", ".join(set_members)
        chips = _chips_html(pred.pred_set, taxa_map, getattr(pred, "probs", None), taxa_by_idx=taxa_by_idx)
        traits_html = _traits_html(pred)
        evidence_html = _evidence_html(pred)
        rationale_html = _rationale_html(pred)

        items = []
        for idx in pred.pred_set:
            name = taxa_map.get(idx, f"Taxon {idx}")
            t = taxa_by_idx.get(idx)
            cname = get_common_name(name, gbif_key=getattr(t, "gbif_key", None) if t else None)
            formatted_name = format_species_name(name, cname)
            prob = float(pred.probs[idx]) if (hasattr(pred, "probs") and idx < len(pred.probs)) else 0.0
            items.append(f"<li><strong>{formatted_name}</strong>: {prob * 100:.1f}%</li>")
        member_list_html = (
            f'<div style="margin:8px 0">'
            f'<div style="font-weight:600;color:#555;margin-bottom:4px">Prediction set members &amp; probabilities:</div>'
            f'<ul style="margin:4px 0 8px 20px;padding:0;color:#1a1a1a">{"".join(items)}</ul>'
            f'</div>'
        )

        rank_info = ""
        if pred.rank in ("genus", "family") and pred.label:
            rank_info = (
                f'<div style="color:#666;font-size:0.9em;margin-top:4px">'
                f'Closest rank: <strong>{pred.label}</strong> ({pred.rank}, confidence {pred.confidence * 100:.1f}%)'
                f'</div>'
            )

        return (
            '<div style="border:2px solid #ffb74d;border-radius:12px;padding:16px;'
            'background:#fffde7;margin:8px 0;color:#1a1a1a">'
            f'<div style="font-size:1.3em;font-weight:700;color:#e65100;margin-bottom:4px">'
            f'Not sure: {members_str}</div>'
            f'{rank_info}'
            f'{member_list_html}'
            f'{chips}'
            f'{traits_html}'
            f'{evidence_html}'
            f'{rationale_html}'
            '</div>'
        )

    # Case 4: Confident prediction
    gbif_key = None
    if hasattr(pipeline, "taxa"):
        for t in pipeline.taxa:
            if getattr(t, "species", None) == pred.label:
                gbif_key = getattr(t, "gbif_key", None)
                break
    cname = get_common_name(pred.label, gbif_key=gbif_key) if pred.rank == "species" else None
    if cname and cname.strip().lower() != pred.label.strip().lower():
        label_html = (
            f'<span style="font-size:1.3em;font-weight:700;color:#1a1a1a">{cname}</span> '
            f'<span style="font-size:1.0em;font-style:italic;color:#444">({pred.label})</span>'
        )
    else:
        label_html = f'<span style="font-size:1.3em;font-weight:700;color:#1a1a1a">{pred.label}</span>'
    conf_pct = f"{pred.confidence * 100:.1f}%"
    chips = _chips_html(pred.pred_set, taxa_map, getattr(pred, "probs", None), taxa_by_idx=taxa_by_idx) if pred.pred_set else ""
    traits_html = _traits_html(pred)
    evidence_html = _evidence_html(pred)
    rationale_html = _rationale_html(pred)

    return (
        '<div style="border:2px solid #81c784;border-radius:12px;padding:16px;'
        'background:#f9fbe7;margin:8px 0;color:#1a1a1a">'
        f'<div style="margin-bottom:4px">{label_html}</div>'
        f'<div style="color:#2e7d32;font-weight:600">Confidence: {conf_pct}</div>'
        f'{chips}'
        f'{traits_html}'
        f'{evidence_html}'
        f'{rationale_html}'
        '</div>'
    )


def _hint_html() -> str:
    return (
        '<div style="background:#fff3e0;border:1px solid #ffb300;border-radius:8px;'
        'padding:10px;margin-top:8px;color:#1a1a1a">'
        '📸 <strong>Tip:</strong> Adding a flower photo often narrows the species identification.'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Predict callback
# ---------------------------------------------------------------------------

def predict_fn(images, lat_str: str = "", lon_str: str = ""):
    """Core prediction callback for 1-3 photos of one plant observation."""
    if not images:
        return (
            "<p style='color:#c62828'>Please upload at least one photo.</p>",
            "",
            gr.update(visible=False),
        )

    # Build path list
    paths = []
    for img in images:
        if img is None:
            continue
        if isinstance(img, (str, Path)):
            paths.append(str(img))
        elif hasattr(img, "name"):
            paths.append(img.name)
        else:
            paths.append(str(img))

    if not paths:
        return (
            "<p style='color:#c62828'>No valid image files found.</p>",
            "",
            gr.update(visible=False),
        )

    if len(paths) > 3:
        return (
            f"<p style='color:#c62828'>Please upload 1 to 3 photos of the same plant (got {len(paths)}).</p>",
            "",
            gr.update(visible=False),
        )

    # Attempt EXIF GPS from first image if lat/lon not provided
    lat, lon = None, None
    if isinstance(lat_str, str) and lat_str.strip():
        try:
            lat = float(lat_str.strip())
        except ValueError:
            pass
    if isinstance(lon_str, str) and lon_str.strip():
        try:
            lon = float(lon_str.strip())
        except ValueError:
            pass

    if lat is None and lon is None:
        gps = _try_exif_gps(paths[0])
        if gps:
            lat, lon = gps

    # Run prediction via observation helper
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = load_pipeline()
    pipeline = _PIPELINE
    try:
        pred = predict_observation(pipeline, paths, lat=lat, lon=lon)
    except Exception as exc:
        return (
            f"<p style='color:#c62828'>Prediction error: {exc}</p>",
            "",
            gr.update(visible=False),
        )

    result_card_html = _result_html(pred, pipeline)

    # Hint: one photo, multiple species in pred_set, and detected organ is not "flower"
    hint_card_html = ""
    if len(paths) == 1 and len(pred.pred_set) > 1:
        organ = getattr(pred, "organ", None)
        if organ is None or (organ in ORGANS and organ != "flower"):
            hint_card_html = _hint_html()

    show_openset = is_open_set_reject(pred)
    return result_card_html, hint_card_html, gr.update(visible=show_openset)


# ---------------------------------------------------------------------------
# Add-species callback
# ---------------------------------------------------------------------------

def add_species_fn(name: str, genus: str = "", family: str = "", photos = None):
    """Add a new species to the pipeline with name + up to 10 photos (no retraining)."""
    if not name or not name.strip():
        return "⚠️ Species name is required."

    paths = []
    if photos:
        if isinstance(photos, (str, Path)):
            paths.append(str(photos))
        else:
            for img in photos:
                if img is None:
                    continue
                if isinstance(img, (str, Path)):
                    paths.append(str(img))
                elif hasattr(img, "name"):
                    paths.append(img.name)
                else:
                    paths.append(str(img))

    if not paths:
        return "⚠️ Please upload at least 1 reference photo (up to 10 photos)."
    if len(paths) > 10:
        return f"⚠️ Please upload at most 10 photos (got {len(paths)})."

    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = load_pipeline()
    pipeline = _PIPELINE

    sp_name = name.strip()
    parts = sp_name.split()
    g = genus.strip() if genus and genus.strip() else (parts[0] if parts else "")
    f = family.strip() if family and family.strip() else ""
    taxon = Taxon(idx=-1, species=sp_name, genus=g, family=f)
    try:
        pipeline.add_species(taxon, paths)
    except Exception as exc:
        return f"❌ Error: {exc}"

    photo_word = "photo" if len(paths) == 1 else "photos"
    return f"✅ Added **{sp_name}** to the pipeline ({len(paths)} {photo_word}, no retraining)."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

MOBILE_CSS = """
/* Mobile-friendly single-column layout below 700 px. No external CDN needed. */
body { font-family: system-ui, sans-serif; }

@media (max-width: 700px) {
  .gradio-container { padding: 8px !important; }
  .gr-row { flex-direction: column !important; }
  .gr-column { min-width: 100% !important; max-width: 100% !important; }
  .upload-container { min-height: 160px !important; }
  #result-card, #hint-row { font-size: 0.95em; }
}

#result-card { min-height: 60px; }
#result-card > div, #hint-row > div { color: #1a1a1a; }
#result-card li, #result-card strong, #result-card p, #hint-row strong { color: inherit; }
"""


def build_ui() -> gr.Blocks:
    global _PIPELINE
    _PIPELINE = load_pipeline()

    with gr.Blocks(title="VanaLens \u2013 Plant Identifier") as demo:
        gr.Markdown(
            "# 🌿 VanaLens – Plant Identifier\n"
            "Upload 1–3 photos of the **same plant** (JPEG/PNG) for instant identification."
        )

        with gr.Tabs():
            # ----------------------------------------------------------------
            # Tab 1: Identify
            # ----------------------------------------------------------------
            with gr.Tab("Identify"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_upload = gr.File(
                            label="Plant photos (1–3, JPEG/PNG)",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        with gr.Row():
                            lat_field = gr.Textbox(
                                label="Latitude (optional)",
                                placeholder="e.g. 12.9716",
                                scale=1,
                            )
                            lon_field = gr.Textbox(
                                label="Longitude (optional)",
                                placeholder="e.g. 77.5946",
                                scale=1,
                            )
                        identify_btn = gr.Button("🔍 Identify", variant="primary")

                    with gr.Column(scale=1):
                        result_html = gr.HTML(elem_id="result-card", label="Result")
                        hint_html = gr.HTML(elem_id="hint-row")
                        with gr.Accordion("🌱 Add this species to catalog", open=True, visible=False) as add_openset_box:
                            gr.Markdown(
                                "**Not in catalog?** Add this species with its name and up to 10 photos.\n"
                                "*(Few-shot addition – no retraining needed)*"
                            )
                            openset_name = gr.Textbox(label="Species name", placeholder="e.g. Ficus elastica")
                            openset_photos = gr.File(
                                label="Reference photos (up to 10)",
                                file_count="multiple",
                                file_types=["image"],
                            )
                            openset_btn = gr.Button("➕ Add to catalog", variant="primary")
                            openset_status = gr.Markdown("")

                            def add_from_openset(name, photos, orig_photos):
                                p = photos if photos else orig_photos
                                return add_species_fn(name=name, genus="", family="", photos=p)

                            openset_btn.click(
                                fn=add_from_openset,
                                inputs=[openset_name, openset_photos, image_upload],
                                outputs=[openset_status],
                            )

                identify_btn.click(
                    fn=predict_fn,
                    inputs=[image_upload, lat_field, lon_field],
                    outputs=[result_html, hint_html, add_openset_box],
                )

                # Auto-fill GPS from EXIF when first image is uploaded
                def autofill_gps(files):
                    if not files:
                        return "", ""
                    path = files[0] if isinstance(files[0], str) else getattr(files[0], "name", None)
                    if path:
                        gps = _try_exif_gps(path)
                        if gps:
                            return str(gps[0]), str(gps[1])
                    return "", ""

                image_upload.change(
                    fn=autofill_gps,
                    inputs=[image_upload],
                    outputs=[lat_field, lon_field],
                )

            # ----------------------------------------------------------------
            # Tab 2: Add a species
            # ----------------------------------------------------------------
            with gr.Tab("Add a species"):
                gr.Markdown(
                    "Add a new species to the pipeline with up to 10 reference photos.\n"
                    "*(No retraining – few-shot kNN addition.)*"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        sp_name = gr.Textbox(label="Species name", placeholder="e.g. Tectona grandis")
                        sp_genus = gr.Textbox(label="Genus (optional)", placeholder="e.g. Tectona")
                        sp_family = gr.Textbox(label="Family (optional)", placeholder="e.g. Lamiaceae")
                        sp_photos = gr.File(
                            label="Reference photos (up to 10)",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        add_btn = gr.Button("➕ Add species", variant="primary")
                    with gr.Column(scale=1):
                        add_status = gr.Markdown("")

                add_btn.click(
                    fn=add_species_fn,
                    inputs=[sp_name, sp_genus, sp_family, sp_photos],
                    outputs=[add_status],
                )

    return demo


if __name__ == "__main__":
    build_ui().launch(css=MOBILE_CSS)
