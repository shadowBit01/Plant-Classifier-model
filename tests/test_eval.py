"""Lane L1 tests: metrics, harness, data utilities."""
from __future__ import annotations

import numpy as np
from PIL import Image

from eval.harness import evaluate
from eval.metrics import auroc, ece, macro_top1, rank_acc, topk_acc
from vanalens.contracts import Sample, Taxon, load_taxa, save_taxa
from vanalens.data import (
    cap_per_class,
    grouped_split,
    load_inat_india,
    read_exif_gps,
    read_manifest,
    write_manifest,
)

P3 = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1]])
Y3 = np.array([0, 1, 1])


def test_topk_and_macro():
    assert abs(topk_acc(P3, Y3, 1) - 1 / 3) < 1e-9
    assert topk_acc(P3, Y3, 2) == 1.0
    assert abs(macro_top1(P3, Y3) - 0.5) < 1e-9


def test_ece_perfect_is_zero():
    y = np.array([0, 1, 2, 1])
    assert ece(np.eye(3)[y], y) < 1e-9


def test_rank_acc(taxa):
    P = np.array([[0.1, 0.5, 0.1, 0.2, 0.1]])      # predicts species 1 (Alpha two)
    assert rank_acc(P, np.array([0]), taxa, "genus") == 1.0   # true species 0 is also Alpha
    assert rank_acc(P, np.array([3]), taxa, "genus") == 0.0


def test_auroc():
    assert abs(auroc(np.array([0.9, 0.8]), np.array([0.1, 0.85])) - 0.75) < 1e-9


def test_evaluate_keys(taxa, calibrated):
    P, y = calibrated
    r = evaluate(P[:100], y[:100], taxa, sets=[np.array([int(i)]) for i in P[:100].argmax(1)])
    for k in ("n", "top1", "top5", "macro_top1", "ece", "genus_top1", "family_top1", "coverage", "mean_set_size"):
        assert k in r
    assert r["n"] == 100 and r["mean_set_size"] == 1.0


def _samples(n_groups=30, per_group=3, n_cls=3):
    out = []
    for g in range(n_groups):
        for i in range(per_group):
            out.append(Sample(f"s/{g}/{i}", f"/x/{g}_{i}.jpg", label=g % n_cls, source="t", group=f"g{g}"))
    return out


def test_grouped_split_no_leak():
    sp = grouped_split(_samples(), seed=0)
    assert set(sp) == {"gallery", "cal", "test"}
    groups = [{s.group for s in sp[k]} for k in ("gallery", "cal", "test")]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
    assert sum(len(v) for v in sp.values()) == 90
    for k in sp:
        assert {s.label for s in sp[k]} == {0, 1, 2}


def test_cap_per_class():
    capped = cap_per_class(_samples(), cap=10, seed=0)
    counts = np.bincount([s.label for s in capped])
    assert counts.max() <= 10
    assert [s.image_id for s in capped] == [s.image_id for s in cap_per_class(_samples(), cap=10, seed=0)]


def test_manifest_roundtrip(tmp_path):
    s = _samples(2, 2)
    s[0].lat, s[0].lon, s[0].organ = 23.17, 79.93, "leaf"
    write_manifest(s, tmp_path / "m.jsonl")
    assert read_manifest(tmp_path / "m.jsonl") == s


def test_exif_none_without_gps(tmp_path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(p)
    assert read_exif_gps(str(p)) is None


def test_load_inat_india(tmp_path):
    sp_dir = tmp_path / "Ficus religiosa"
    sp_dir.mkdir(parents=True)
    img_path = sp_dir / "1001.jpg"
    Image.new("RGB", (8, 8)).save(img_path)
    save_taxa([Taxon(0, "Ficus religiosa", "Ficus", "Moraceae", 5361909)], tmp_path / "labels.json")
    (tmp_path / "meta.csv").write_text(
        "image,lat,lon,obs_id,observed_on,license,url\n"
        "Ficus religiosa/1001.jpg,23.18,79.99,1001,2025-08-01,cc-by,https://example.com/1001.jpg\n",
        encoding="utf-8",
    )
    samples, taxa = load_inat_india(tmp_path)
    assert len(taxa) == 1 and taxa[0].species == "Ficus religiosa" and taxa[0].genus == "Ficus"
    assert len(samples) == 1
    assert samples[0].source == "inat"
    assert samples[0].group == "1001"
    assert samples[0].lat == 23.18 and samples[0].lon == 79.99
    assert load_taxa(tmp_path / "labels.json") == taxa
