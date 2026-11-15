"""Offline tests for vl600 dataset builder, loader, manifests, stats, and campus mapping."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from eval.build_vl600 import (
    S3_BASE,
    build_vl600,
    check_disk_budget,
    clean_binomial,
    fetch_photos_for_species,
    s3_medium_url,
    select_vl600_species,
    write_species_csv,
)
from scripts.make_manifests import _process_campus, _process_vl600
from vanalens.contracts import Taxon, load_taxa, save_taxa
from vanalens.data import load_campus, load_vl600, read_manifest


def test_clean_binomial_and_s3_url():
    assert clean_binomial("Hibiscus × rosa-sinensis") == "Hibiscus rosa-sinensis"
    assert clean_binomial("Musa × paradisiaca") == "Musa paradisiaca"
    assert clean_binomial("Azadirachta indica var. minor") == "Azadirachta indica"

    info = s3_medium_url(
        {
            "id": 398092381,
            "license_code": "CC-BY-NC",
            "url": "https://static.inaturalist.org/photos/398092381/square.jpeg",
        }
    )
    assert info == (
        "398092381",
        f"{S3_BASE}/398092381/medium.jpeg",
        "cc-by-nc",
    )
    assert s3_medium_url({"id": 10, "license_code": "cc-by-sa", "url": "x.jpg"}) is None


def test_check_disk_budget():
    logs: list[str] = []
    ok, exp_gb, free_gb = check_disk_budget(50_000, 75_000_000_000, log_fn=logs.append)
    assert ok is True
    assert abs(exp_gb - 4.0) < 1e-6

    # Exceeds 15 GB
    logs.clear()
    ok2, _, _ = check_disk_budget(200_000, 75_000_000_000, log_fn=logs.append)
    assert ok2 is False
    assert any("BLOCKED" in m for m in logs)

    # Leaves under 20 GB free
    logs.clear()
    ok3, _, _ = check_disk_budget(50_000, 22_000_000_000, log_fn=logs.append)
    assert ok3 is False
    assert any("BLOCKED" in m for m in logs)


def test_fetch_photos_caps_and_casual_rule():
    # Build synthetic API responses for a common-list species
    india_obs = [
        {
            "id": 100 + i,
            "observed_on": f"2024-05-{(i % 28) + 1:02d}",
            "quality_grade": "research",
            "place_ids": [6681],
            "geojson": {"coordinates": [77.5, 12.9]},
            "photos": [
                {"id": (100 + i) * 10 + p, "license_code": "cc-by", "url": f"https://x/{p}.jpg"}
                for p in range(4)  # 4 photos -> capped at 3 per obs
            ],
        }
        for i in range(2)
    ]
    world_obs = [
        {
            "id": 200,
            "observed_on": "2025-08-10",
            "quality_grade": "research",
            "place_ids": [1],
            "location": "10.5, 20.5",
            "photos": [{"id": 2001, "license_code": "cc0", "url": "https://x/1.jpeg"}],
        }
    ]
    casual_obs = [
        # Valid casual observation: community_taxon_id matches and >= 1 agreement
        {
            "id": 300,
            "observed_on": "2025-09-01",
            "quality_grade": "casual",
            "community_taxon_id": 999,
            "num_identification_agreements": 2,
            "place_ids": [6681],
            "photos": [{"id": 3001, "license_code": "cc-by-nc", "url": "https://x/c.jpg"}],
        },
        # Invalid casual observation: 0 agreements
        {
            "id": 301,
            "observed_on": "2025-09-02",
            "quality_grade": "needs_id",
            "community_taxon_id": 999,
            "num_identification_agreements": 0,
            "place_ids": [6681],
            "photos": [{"id": 3011, "license_code": "cc-by", "url": "https://x/bad.jpg"}],
        },
    ]

    def fake_api(endpoint: str, params: dict[str, object]) -> dict:
        if params.get("quality_grade") == "research":
            if params.get("place_id") == 6681:
                return {"results": india_obs}
            return {"results": world_obs}
        return {"results": casual_obs}

    sp_info = {
        "name": "Hibiscus rosa-sinensis",
        "taxon_id": 999,
        "india_count": 10,
        "world_count": 20,
        "source": "common",
        "in_common_list": True,
    }
    recs = fetch_photos_for_species(sp_info, api_get=fake_api, max_pre_photos=5, max_post_obs=5)
    pre = [r for r in recs if str(r["observed_on"]) < "2025-07-01"]
    post = [r for r in recs if str(r["observed_on"]) >= "2025-07-01"]
    # Pre-cutoff capped at max_pre_photos=5 (3 from obs 100, 2 from obs 101)
    assert len(pre) == 5
    assert all(r["in_india"] is True for r in pre)
    # Post-cutoff has obs 200 (world RG) and obs 300 (valid casual), but NOT obs 301
    post_obs_ids = {r["obs_id"] for r in post}
    assert post_obs_ids == {"200", "300"}


def test_load_vl600_and_splits_never_drop_species(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VANALENS_OFFLINE", "1")
    vl_root = tmp_path / "vl600"
    vl_root.mkdir(parents=True)

    # 2 species in species.csv: one has photos, one has 0 test photos (never dropped!)
    species_list = [
        {
            "name": "Azadirachta indica",
            "common_name": "Neem",
            "source": "india",
            "count": 500,
            "india_count": 500,
            "world_count": 600,
            "taxon_id": 319135,
        },
        {
            "name": "Rosa webbiana",
            "common_name": "rose",
            "source": "common",
            "count": 50,
            "india_count": 50,
            "world_count": 80,
            "taxon_id": 741910,
        },
    ]
    write_species_csv(species_list, vl_root / "species.csv")

    sp_dir = vl_root / "Azadirachta indica"
    sp_dir.mkdir(parents=True)
    meta_lines = ["image,obs_id,photo_id,lat,lon,observed_on,license,url,quality_grade,in_india"]
    # 5 pre-cutoff observations (2 photos each for obs 1) and 2 post-cutoff observations
    for oid in range(1, 6):
        for pid in (1, 2) if oid == 1 else (1,):
            fname = f"{oid}_{pid}.jpg"
            Image.new("RGB", (8, 8)).save(sp_dir / fname)
            meta_lines.append(
                f"Azadirachta indica/{fname},{oid},{pid},23.1,79.9,2024-03-0{oid},cc-by,{S3_BASE}/{pid}/medium.jpg,research,True"
            )
    for oid in (10, 11):
        fname = f"{oid}_1.jpg"
        Image.new("RGB", (8, 8)).save(sp_dir / fname)
        meta_lines.append(
            f"Azadirachta indica/{fname},{oid},1,23.2,79.8,2025-08-{oid},cc0,{S3_BASE}/1/medium.jpg,research,True"
        )
    (vl_root / "meta.csv").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    samples, taxa = load_vl600(vl_root)
    assert [t.species for t in taxa] == ["Azadirachta indica", "Rosa webbiana"]
    assert len(samples) == 8

    manifests_dir = tmp_path / "manifests"
    info = _process_vl600(vl_root, manifests_dir)
    assert info["species"] == 2
    assert info["species_lt5_test_obs"] == 2

    gal = read_manifest(manifests_dir / "vl600_gallery.jsonl")
    cal = read_manifest(manifests_dir / "vl600_cal.jsonl")
    tst = read_manifest(manifests_dir / "vl600_test.jsonl")
    # Ensure obs 1's two photos stay together in the same split
    gal_groups = {s.group for s in gal}
    cal_groups = {s.group for s in cal}
    tst_groups = {s.group for s in tst}
    assert not (gal_groups & cal_groups)
    assert not (gal_groups & tst_groups)
    assert not (cal_groups & tst_groups)
    assert tst_groups == {"10", "11"}


def test_load_campus_and_open_set_manifest(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VANALENS_OFFLINE", "1")
    vl_taxa = [
        Taxon(0, "Azadirachta indica", "Azadirachta", "Meliaceae", 3190474),
        Taxon(1, "Ficus religiosa", "Ficus", "Moraceae", 5361909),
    ]
    campus_root = tmp_path / "campus"
    (campus_root / "Ficus religiosa").mkdir(parents=True)
    (campus_root / "Unknown exotica").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(campus_root / "Ficus religiosa" / "p1.jpg")
    Image.new("RGB", (8, 8)).save(campus_root / "Unknown exotica" / "u1.jpg")
    (campus_root / "meta.csv").write_text(
        "image,lat,lon,organ,plant_id\n"
        "Ficus religiosa/p1.jpg,23.18,79.98,leaf,tree_7\n",
        encoding="utf-8",
    )

    manifests_dir = tmp_path / "manifests"
    res = _process_campus(campus_root, manifests_dir, vl600_taxa=vl_taxa)
    assert (manifests_dir / "campus_test.jsonl").exists()
    assert not (manifests_dir / "campus_gallery.jsonl").exists()
    assert not (manifests_dir / "campus_cal.jsonl").exists()
    assert res["open_set_species"] == ["Unknown exotica"]

    loaded = read_manifest(manifests_dir / "campus_test.jsonl")
    by_sp = {s.image_id.split("/")[1]: s for s in loaded}
    assert by_sp["Ficus religiosa"].label == 1
    assert by_sp["Ficus religiosa"].group == "tree_7"
    assert by_sp["Ficus religiosa"].organ == "leaf"
    assert by_sp["Ficus religiosa"].lat == 23.18
    assert by_sp["Unknown exotica"].label is None


def test_build_vl600_end_to_end_offline(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VANALENS_OFFLINE", "1")

    def fake_api(endpoint: str, params: dict[str, object]) -> dict:
        if endpoint == "observations/species_counts":
            if params.get("place_id") == 6681:
                return {
                    "total_results": 2,
                    "results": [
                        {
                            "count": 300,
                            "taxon": {
                                "id": 101,
                                "name": "Azadirachta indica",
                                "preferred_common_name": "Neem",
                                "observations_count": 500,
                            },
                        },
                        {
                            "count": 200,
                            "taxon": {
                                "id": 102,
                                "name": "Ficus religiosa",
                                "preferred_common_name": "Peepal",
                                "observations_count": 400,
                            },
                        },
                    ],
                }
            return {
                "total_results": 1,
                "results": [
                    {
                        "count": 1000,
                        "taxon": {
                            "id": 201,
                            "name": "Quercus robur",
                            "preferred_common_name": "English Oak",
                            "observations_count": 1000,
                        },
                    }
                ],
            }
        if endpoint == "taxa":
            q = str(params.get("q") or "Rosa")
            name = q if len(q.split()) == 2 else f"{q} indica"
            return {
                "results": [
                    {
                        "id": 900 + abs(hash(name)) % 100,
                        "name": name,
                        "preferred_common_name": q,
                        "observations_count": 50,
                    }
                ]
            }
        if endpoint == "observations":
            return {
                "results": [
                    {
                        "id": 5001,
                        "observed_on": "2024-04-10",
                        "quality_grade": "research",
                        "place_ids": [6681],
                        "photos": [{"id": 7001, "license_code": "cc0", "url": "https://x/7001.jpg"}],
                    },
                    {
                        "id": 5002,
                        "observed_on": "2025-08-10",
                        "quality_grade": "research",
                        "place_ids": [6681],
                        "photos": [{"id": 7002, "license_code": "cc-by", "url": "https://x/7002.jpg"}],
                    },
                ]
            }
        return {"results": []}

    def fake_download(task: tuple[Path, str]) -> bool:
        dest, url = task
        assert url.startswith(S3_BASE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(dest, format="JPEG")
        return True

    stats = build_vl600(
        root=tmp_path / "data" / "vl600",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        api_get=fake_api,
        download_fn=fake_download,
        free_bytes_fn=lambda _: 80_000_000_000,
        n_india=2,
        n_world=1,
    )
    assert stats["coverage_pct"] == 100.0
    assert (tmp_path / "data" / "vl600" / "species.csv").exists()
    assert (tmp_path / "data" / "vl600" / "meta.csv").exists()
    assert (tmp_path / "data" / "vl600" / "labels.json").exists()
    assert (tmp_path / "runs" / "L1" / "vl600_stats.json").exists()
    saved_stats = json.loads((tmp_path / "runs" / "L1" / "vl600_stats.json").read_text(encoding="utf-8"))
    assert saved_stats["species"] == stats["species"]
