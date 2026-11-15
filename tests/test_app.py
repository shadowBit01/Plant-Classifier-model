"""Lane L5 tests: the UI builds and the mock pipeline honours the Prediction contract."""
from __future__ import annotations

import numpy as np
import pytest

from vanalens.contracts import Prediction, RANKS, Taxon
from vanalens.pipeline import MockPipeline
from app.app import (
    MOBILE_CSS,
    _COMMON_NAMES_CACHE,
    add_species_fn,
    average_log_probabilities,
    format_species_name,
    get_common_name,
    is_confident,
    is_open_set_reject,
    load_common_names,
    predict_fn,
    predict_observation,
    _chips_html,
    _evidence_html,
    _hint_html,
    _result_html,
    _traits_html,
)


def test_mock_prediction_contract():
    m = MockPipeline()
    for paths in (["a.jpg"], ["leaf.jpg", "flower.jpg"]):
        p = m.predict(paths)
        assert abs(p.probs.sum() - 1) < 1e-9 and p.pred_set and p.rank in RANKS
        assert p.top_k[0][0] == int(np.argmax(p.probs)) and len(p.evidence) == 3


def test_mock_add_species():
    m = MockPipeline()
    m.add_species(Taxon(-1, "Tectona grandis", "Tectona", "Lamiaceae"), ["x.jpg"])
    assert m.taxa[-1].idx == 5


def test_ui_builds(monkeypatch):
    pytest.importorskip("gradio")
    monkeypatch.setenv("VANALENS_MOCK", "1")
    from app.app import build_ui
    assert build_ui() is not None


def test_space_files_and_front_matter():
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    readme = (repo / "space" / "README.md").read_text(encoding="utf-8")
    reqs = (repo / "space" / "requirements.txt").read_text(encoding="utf-8")
    entry = repo / "space" / "app.py"
    assert entry.is_file()
    assert readme.startswith("---\n") and "\n---\n" in readme[4:]
    front_matter = readme.split("---", 2)[1]
    assert "sdk: gradio" in front_matter
    assert "app_file: app.py" in front_matter
    assert "gradio" in reqs and "faiss-cpu" in reqs


def test_space_loads_state_from_env_cache(tmp_path, monkeypatch, taxa, clusters):
    import json
    from vanalens import config as vanalens_config
    from vanalens.contracts import save_taxa
    from vanalens.pipeline import VanaLens

    cache_dir = tmp_path / "custom_cache"
    (cache_dir / "state").mkdir(parents=True)
    (cache_dir / "emb").mkdir(parents=True)
    (cache_dir / "inat_india").mkdir(parents=True)

    state = {
        "scorers": ["knn", "probe"],
        "weights": {"knn": 0.5, "probe": 0.5},
        "T": 0.85,
        "qhat": 0.42,
        "thr": 0.25,
        "beta": 0.1,
    }
    (cache_dir / "state" / "inat_india.json").write_text(json.dumps(state), encoding="utf-8")
    clusters["gallery"].save(cache_dir / "emb" / "inat_india_gallery.npz")
    save_taxa(taxa, cache_dir / "inat_india" / "labels.json")

    monkeypatch.delenv("VANALENS_MOCK", raising=False)
    monkeypatch.delenv("VANALENS_DATA", raising=False)
    monkeypatch.setenv("VANALENS_CACHE", str(cache_dir))
    monkeypatch.setattr(vanalens_config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(vanalens_config, "DATA_DIR", tmp_path / "nonexistent_data")

    import os
    import space.app as space_app

    space_app.configure_space_env()
    assert os.environ["VANALENS_MOCK"] == "0"
    pipeline = space_app.load_pipeline()
    assert isinstance(pipeline, VanaLens)
    assert abs(pipeline.T - 0.85) < 1e-6 and abs(pipeline.qhat - 0.42) < 1e-6
    assert len(pipeline.taxa) == len(taxa)


def test_observation_predict_1_to_3_photos():
    m = MockPipeline()
    # 1 photo
    p1 = predict_observation(m, ["leaf.jpg"])
    assert isinstance(p1, Prediction)
    # 2 photos
    p2 = predict_observation(m, ["leaf.jpg", "flower.jpg"])
    assert isinstance(p2, Prediction)
    # 3 photos
    p3 = predict_observation(m, ["leaf.jpg", "flower.jpg", "bark.jpg"])
    assert isinstance(p3, Prediction)

    # >3 photos raises ValueError
    with pytest.raises(ValueError, match="1 to 3 photos"):
        predict_observation(m, ["1.jpg", "2.jpg", "3.jpg", "4.jpg"])


def test_single_photo_pipeline_fallback_averages_log_probs():
    class SinglePhotoMock:
        def __init__(self):
            self.taxa = [
                Taxon(0, "Species A", "GenusA", "FamilyA"),
                Taxon(1, "Species B", "GenusA", "FamilyA"),
            ]

        def predict(self, path: str, lat=None, lon=None, use_rerank=True):
            if isinstance(path, list):
                raise TypeError("SinglePhotoMock only accepts a single photo string")
            probs = np.array([0.7, 0.3]) if "photo1" in path else np.array([0.4, 0.6])
            return Prediction(
                probs=probs,
                top_k=[(0, float(probs[0])), (1, float(probs[1]))],
                pred_set=[0, 1],
                rank="genus",
                label="GenusA",
                confidence=1.0,
                evidence=["ref1.jpg"],
                traits=["trait1"],
                organ="leaf",
            )

    pipe = SinglePhotoMock()
    # Should catch TypeError and call average_log_probabilities
    combined = predict_observation(pipe, ["photo1.jpg", "photo2.jpg"])
    assert isinstance(combined, Prediction)
    # Log-probs: ln([0.7, 0.3]) and ln([0.4, 0.6]) averaged
    expected_logp = 0.5 * (np.log([0.7, 0.3]) + np.log([0.4, 0.6]))
    expected_p = np.exp(expected_logp - expected_logp.max())
    expected_p /= expected_p.sum()
    assert np.allclose(combined.probs, expected_p, atol=1e-5)


def test_not_confident_display():
    m = MockPipeline()
    # Case A: conformal set has more than 1 species
    pred_mult = Prediction(
        probs=np.array([0.45, 0.35, 0.1, 0.05, 0.05]),
        top_k=[(0, 0.45), (1, 0.35), (2, 0.1), (3, 0.05), (4, 0.05)],
        pred_set=[0, 1],
        rank="genus",
        label="Ficus",
        confidence=0.80,
    )
    assert not is_confident(pred_mult, m)
    html_mult = _result_html(pred_mult, m)
    assert "Not sure: " in html_mult
    assert "Ficus religiosa" in html_mult
    assert "Ficus benghalensis" in html_mult
    assert "45.0%" in html_mult
    assert "35.0%" in html_mult

    # Case B: conformal set has 1 species, but top prob below state threshold (tau = 0.6 default)
    pred_low_prob = Prediction(
        probs=np.array([0.55, 0.15, 0.1, 0.1, 0.1]),
        top_k=[(0, 0.55), (1, 0.15), (2, 0.1), (3, 0.1), (4, 0.1)],
        pred_set=[0],
        rank="genus",
        label="Ficus",
        confidence=0.55,
    )
    assert not is_confident(pred_low_prob, m)
    html_low = _result_html(pred_low_prob, m)
    assert "Not sure: " in html_low
    assert "Ficus religiosa" in html_low
    assert "55.0%" in html_low

    # Case C: Confident prediction (1 species in set, prob >= threshold)
    pred_conf = Prediction(
        probs=np.array([0.85, 0.05, 0.05, 0.03, 0.02]),
        top_k=[(0, 0.85), (1, 0.05)],
        pred_set=[0],
        rank="species",
        label="Ficus religiosa",
        confidence=0.85,
    )
    assert is_confident(pred_conf, m)
    html_conf = _result_html(pred_conf, m)
    assert "Not sure: " not in html_conf
    assert "Ficus religiosa" in html_conf
    assert "Confidence: 85.0%" in html_conf


def test_openset_reject_display_and_add_species(monkeypatch):
    m = MockPipeline()
    monkeypatch.setattr("app.app._PIPELINE", m)

    # Open-set reject prediction
    pred_openset = Prediction(
        probs=np.array([0.3, 0.25, 0.2, 0.15, 0.1]),
        top_k=[(0, 0.3), (1, 0.25)],
        pred_set=[0, 1],
        rank="unknown",
        label="",
        confidence=0.3,
        rationale="open-set gate: best gallery similarity 0.412 < 1.413 (species not in gallery?)",
    )
    assert is_open_set_reject(pred_openset)
    html = _result_html(pred_openset, m)
    assert "Not in catalog, closest genus/family Ficus" in html
    assert "Add this species" in html

    # Test predict_fn returns visible=True for open-set box
    class MockOpenSetPipeline(MockPipeline):
        def predict(self, paths, lat=None, lon=None, use_rerank=True):
            p = super().predict(paths, lat=lat, lon=lon, use_rerank=use_rerank)
            p.rank = "unknown"
            p.label = ""
            p.rationale = "open-set gate: best gallery similarity 0.350 < 1.413 (species not in gallery?)"
            return p

    monkeypatch.setattr("app.app._PIPELINE", MockOpenSetPipeline())
    card, hint, add_box = predict_fn(["unknown_plant.jpg"])
    assert "Not in catalog, closest genus/family" in card
    # Gradio update dict or Component
    visible_val = add_box.get("visible") if isinstance(add_box, dict) else getattr(add_box, "visible", True)
    assert visible_val is True


def test_add_species_fn_validation_and_execution(monkeypatch):
    m = MockPipeline()
    monkeypatch.setattr("app.app._PIPELINE", m)

    # Missing name
    err1 = add_species_fn("", photos=["p1.jpg"])
    assert "Species name is required" in err1

    # Missing photos
    err2 = add_species_fn("Ficus elastica", photos=[])
    assert "at least 1 reference photo" in err2

    # > 10 photos
    err3 = add_species_fn("Ficus elastica", photos=[f"p{i}.jpg" for i in range(11)])
    assert "at most 10 photos" in err3

    # Valid: 1 photo (up to 10 photos supported)
    res1 = add_species_fn("Ficus elastica", photos=["p1.jpg"])
    assert "Added **Ficus elastica**" in res1
    assert m.taxa[-1].species == "Ficus elastica"
    assert m.taxa[-1].genus == "Ficus"

    # Valid: 5 photos
    res5 = add_species_fn("Santalum album", photos=[f"s{i}.jpg" for i in range(5)])
    assert "Added **Santalum album**" in res5
    assert m.taxa[-1].species == "Santalum album"


def test_predict_fn_photo_count_limits(monkeypatch):
    m = MockPipeline()
    monkeypatch.setattr("app.app._PIPELINE", m)

    # 0 photos
    card, _, _ = predict_fn([])
    assert "upload at least one photo" in card

    # >3 photos
    card, _, _ = predict_fn(["1.jpg", "2.jpg", "3.jpg", "4.jpg"])
    assert "1 to 3 photos" in card

    # 1 to 3 photos succeeds
    card1, _, _ = predict_fn(["1.jpg"])
    assert "Prediction error" not in card1
    card3, _, _ = predict_fn(["1.jpg", "2.jpg", "3.jpg"])
    assert "Prediction error" not in card3


def test_dark_mode_contrast_colors():
    m = MockPipeline()

    # 1. _result_html outer card divs, labels, lists
    # Case 1: Organ gate reject
    pred_organ = Prediction(
        probs=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        top_k=[(0, 1.0)],
        pred_set=[0],
        rank="species",
        label="Unknown",
        confidence=1.0,
        organ="other",
        rationale="organ gate reject",
    )
    html_organ = _result_html(pred_organ, m)
    assert '<div style="border:2px solid #e57373;border-radius:12px;padding:16px;background:#ffebee;margin:8px 0;color:#1a1a1a">' in html_organ

    # Case 2: Open-set reject
    pred_openset = Prediction(
        probs=np.array([0.3, 0.25, 0.2, 0.15, 0.1]),
        top_k=[(0, 0.3), (1, 0.25)],
        pred_set=[0, 1],
        rank="unknown",
        label="",
        confidence=0.3,
        rationale="open-set gate: best gallery similarity 0.412 < 1.413",
    )
    html_openset = _result_html(pred_openset, m)
    assert '<div style="border:2px solid #ef5350;border-radius:12px;padding:16px;background:#fff8e1;margin:8px 0;color:#1a1a1a">' in html_openset

    # Case 3: Not confident
    pred_not_conf = Prediction(
        probs=np.array([0.45, 0.35, 0.1, 0.05, 0.05]),
        top_k=[(0, 0.45), (1, 0.35)],
        pred_set=[0, 1],
        rank="genus",
        label="Ficus",
        confidence=0.80,
    )
    html_not_conf = _result_html(pred_not_conf, m)
    assert '<div style="border:2px solid #ffb74d;border-radius:12px;padding:16px;background:#fffde7;margin:8px 0;color:#1a1a1a">' in html_not_conf
    assert '<ul style="margin:4px 0 8px 20px;padding:0;color:#1a1a1a">' in html_not_conf

    # Case 4: Confident
    pred_conf = Prediction(
        probs=np.array([0.85, 0.05, 0.05, 0.03, 0.02]),
        top_k=[(0, 0.85), (1, 0.05)],
        pred_set=[0],
        rank="species",
        label="Ficus religiosa",
        confidence=0.85,
    )
    html_conf = _result_html(pred_conf, m)
    assert '<div style="border:2px solid #81c784;border-radius:12px;padding:16px;background:#f9fbe7;margin:8px 0;color:#1a1a1a">' in html_conf
    assert "color:#1a1a1a" in html_conf
    assert "color:#444" in html_conf
    assert "Ficus religiosa" in html_conf

    # 2. _chips_html species chip spans
    taxa_map = {0: "Ficus religiosa", 1: "Ficus benghalensis"}
    chips_html = _chips_html([0, 1], taxa_map, np.array([0.6, 0.4]))
    assert "color:#1b5e20" in chips_html
    assert 'border-radius:16px;padding:2px 10px;margin:2px 3px;font-size:0.85em;color:#1b5e20' in chips_html

    # 3. _traits_html ul
    pred_with_traits = Prediction(
        probs=np.array([1.0]),
        top_k=[(0, 1.0)],
        pred_set=[0],
        rank="species",
        label="Ficus religiosa",
        confidence=1.0,
        traits=["broad leaves", "milky sap"],
    )
    traits_html = _traits_html(pred_with_traits)
    assert "<ul style='margin:4px 0 0 16px;padding:0;color:#1a1a1a'>" in traits_html

    # 4. _evidence_html wrapper div and reference spans
    pred_with_evidence = Prediction(
        probs=np.array([1.0]),
        top_k=[(0, 1.0)],
        pred_set=[0],
        rank="species",
        label="Ficus religiosa",
        confidence=1.0,
        evidence=["ref1.jpg", "ref2.jpg"],
    )
    evidence_html = _evidence_html(pred_with_evidence)
    assert '<div style="margin-top:8px;color:#1a1a1a"><strong>Nearest references:</strong>' in evidence_html
    assert 'font-family:monospace;color:#1a1a1a">ref1.jpg</span>' in evidence_html

    # 5. _hint_html div
    hint_html = _hint_html()
    assert '<div style="background:#fff3e0;border:1px solid #ffb300;border-radius:8px;padding:10px;margin-top:8px;color:#1a1a1a">' in hint_html

    # 6. MOBILE_CSS rules
    assert '#result-card > div, #hint-row > div { color: #1a1a1a; }' in MOBILE_CSS
    assert '#result-card li, #result-card strong, #result-card p, #hint-row strong { color: inherit; }' in MOBILE_CSS


def test_format_species_name():
    # Common name exists and differs from species
    assert format_species_name("Ficus religiosa", "Sacred Fig") == "Sacred Fig (Ficus religiosa)"
    assert format_species_name("Azadirachta indica", "Neem") == "Neem (Azadirachta indica)"

    # Common name is identical to species (case-insensitive) -> returns species
    assert format_species_name("Ficus religiosa", "Ficus religiosa") == "Ficus religiosa"
    assert format_species_name("Ficus religiosa", "ficus religiosa") == "Ficus religiosa"

    # Common name is None or empty string -> returns species
    assert format_species_name("Ficus religiosa", None) == "Ficus religiosa"
    assert format_species_name("Ficus religiosa", "") == "Ficus religiosa"
    assert format_species_name("Ficus religiosa", "   ") == "Ficus religiosa"

    # Empty species returns common_name or empty
    assert format_species_name("", "Sacred Fig") == "Sacred Fig"
    assert format_species_name("", "") == ""


def test_load_common_names_from_csv_and_json(tmp_path, monkeypatch):
    import json
    from vanalens import config as vanalens_config

    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    (data_dir / "test_ds").mkdir(parents=True)
    (cache_dir / "test_ds").mkdir(parents=True)

    # 1. DATA_DIR/test_ds/species.csv
    (data_dir / "test_ds" / "species.csv").write_text(
        "name,common_name,count\n"
        "Mangifera indica,Mango,100\n",
        encoding="utf-8",
    )

    # 2. CACHE_DIR/test_ds/species.csv
    (cache_dir / "test_ds" / "species.csv").write_text(
        "species,preferred_common_name,source\n"
        "Azadirachta indica,Neem,inat\n",
        encoding="utf-8",
    )

    # 3. CACHE_DIR/common_names.json
    (cache_dir / "common_names.json").write_text(
        json.dumps({"Ocimum sanctum": "Holy Basil", "Ficus religiosa": "Peepal"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(vanalens_config, "DATA_DIR", data_dir)
    monkeypatch.setattr(vanalens_config, "CACHE_DIR", cache_dir)
    _COMMON_NAMES_CACHE.clear()

    loaded = load_common_names("test_ds")
    assert loaded["Mangifera indica"] == "Mango"
    assert loaded["Azadirachta indica"] == "Neem"
    assert loaded["Ocimum sanctum"] == "Holy Basil"
    assert loaded["Ficus religiosa"] == "Peepal"

    # get_common_name retrieves from cache
    assert get_common_name("Mangifera indica") == "Mango"
    assert get_common_name("Azadirachta indica") == "Neem"
    assert get_common_name("Ocimum sanctum") == "Holy Basil"


def test_get_common_name_inat_api_mock(tmp_path, monkeypatch):
    import json
    from vanalens import config as vanalens_config

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(vanalens_config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(vanalens_config, "DATA_DIR", tmp_path / "data")
    _COMMON_NAMES_CACHE.clear()

    calls = []

    class MockResponse:
        status_code = 200

        def json(self):
            return {
                "total_results": 1,
                "results": [
                    {
                        "id": 48226,
                        "name": "Delonix regia",
                        "preferred_common_name": "Royal Poinciana",
                        "english_common_name": "Flame Tree",
                    }
                ],
            }

    def mock_get(url, params=None, timeout=None, headers=None):
        calls.append((url, params, timeout))
        return MockResponse()

    monkeypatch.setattr("requests.get", mock_get)

    cname = get_common_name("Delonix regia", fetch_api=True)
    assert cname == "Royal Poinciana"
    assert len(calls) == 1
    assert "api.inaturalist.org" in calls[0][0]
    assert calls[0][1]["q"] == "Delonix regia"
    assert calls[0][1]["preferred_place_id"] == 6681
    assert calls[0][2] == 3.0

    # Cached in memory
    assert _COMMON_NAMES_CACHE.get("Delonix regia") == "Royal Poinciana"

    # Cached in CACHE_DIR/common_names.json
    cache_file = cache_dir / "common_names.json"
    assert cache_file.is_file()
    cached_json = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached_json.get("Delonix regia") == "Royal Poinciana"


def test_get_common_name_gbif_fallback_mock(tmp_path, monkeypatch):
    from vanalens import config as vanalens_config

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(vanalens_config, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(vanalens_config, "DATA_DIR", tmp_path / "data")
    _COMMON_NAMES_CACHE.clear()

    calls = []

    def mock_get(url, params=None, timeout=None, headers=None):
        calls.append((url, params))

        class MockResp:
            status_code = 200

            def __init__(self, data):
                self._data = data

            def json(self):
                return self._data

        if "api.inaturalist.org" in url:
            # iNat has no results
            return MockResp({"total_results": 0, "results": []})
        elif "species/match" in url:
            # GBIF match resolves usageKey
            return MockResp({"usageKey": 5352251, "speciesKey": 5352251})
        elif "vernacularNames" in url:
            # GBIF vernacular names, preferring 'eng'
            return MockResp(
                {
                    "results": [
                        {"vernacularName": "Gulmohar", "language": "hin"},
                        {"vernacularName": "Flamboyant", "language": "eng"},
                        {"vernacularName": "Peacock Flower", "language": "eng"},
                    ]
                }
            )
        return MockResp({})

    monkeypatch.setattr("requests.get", mock_get)

    # 1. With gbif_key provided
    cname1 = get_common_name("Delonix regia", gbif_key=5352251, fetch_api=True)
    assert cname1 == "Flamboyant"

    # 2. With gbif_key=None -> resolves usageKey via /v1/species/match
    _COMMON_NAMES_CACHE.clear()
    (cache_dir / "common_names.json").unlink(missing_ok=True)
    cname2 = get_common_name("Delonix regia", gbif_key=None, fetch_api=True)
    assert cname2 == "Flamboyant"
    assert any("species/match" in c[0] for c in calls)


def test_get_common_name_offline_graceful(tmp_path, monkeypatch):
    import requests
    from vanalens import config as vanalens_config

    monkeypatch.setattr(vanalens_config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(vanalens_config, "DATA_DIR", tmp_path / "data")
    _COMMON_NAMES_CACHE.clear()

    def mock_get(url, params=None, timeout=None, headers=None):
        raise requests.exceptions.ConnectTimeout("Connection timed out")

    monkeypatch.setattr("requests.get", mock_get)

    # Must fail gracefully offline and return None
    result = get_common_name("Uncataloged species", fetch_api=True)
    assert result is None


def test_result_html_case3_and_case4_with_common_names(monkeypatch):
    m = MockPipeline()
    _COMMON_NAMES_CACHE.clear()
    _COMMON_NAMES_CACHE["Ficus religiosa"] = "Sacred Fig"
    _COMMON_NAMES_CACHE["Ficus benghalensis"] = "Banyan"

    # Case 4: Confident species shows Common Name prominently (#1a1a1a) and Scientific Name alongside (#444)
    pred_conf = Prediction(
        probs=np.array([0.90, 0.05, 0.05]),
        top_k=[(0, 0.90)],
        pred_set=[0],
        rank="species",
        label="Ficus religiosa",
        confidence=0.90,
    )
    html_conf = _result_html(pred_conf, m)
    assert '<span style="font-size:1.3em;font-weight:700;color:#1a1a1a">Sacred Fig</span>' in html_conf
    assert '<span style="font-size:1.0em;font-style:italic;color:#444">(Ficus religiosa)</span>' in html_conf
    assert "Sacred Fig (Ficus religiosa)" in html_conf
    assert "Ficus religiosa" in html_conf

    # Case 3: Not confident shows both common and scientific names in Not sure:, list, and chips
    pred_not_conf = Prediction(
        probs=np.array([0.48, 0.42, 0.10]),
        top_k=[(0, 0.48), (1, 0.42)],
        pred_set=[0, 1],
        rank="genus",
        label="Ficus",
        confidence=0.90,
    )
    html_not_conf = _result_html(pred_not_conf, m)
    assert "Not sure: Sacred Fig (Ficus religiosa), Banyan (Ficus benghalensis)" in html_not_conf
    assert "<li><strong>Sacred Fig (Ficus religiosa)</strong>: 48.0%</li>" in html_not_conf
    assert "<li><strong>Banyan (Ficus benghalensis)</strong>: 42.0%</li>" in html_not_conf
    assert "Sacred Fig (Ficus religiosa) (48.0%)" in html_not_conf
    assert "Banyan (Ficus benghalensis) (42.0%)" in html_not_conf
    # Scientific names still present in the HTML
    assert "Ficus religiosa" in html_not_conf
    assert "Ficus benghalensis" in html_not_conf


