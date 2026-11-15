"""Unit tests for Lane L2 baseline B2 (scripts/baseline_bioclip.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import baseline_bioclip
from vanalens import contracts, data


def test_baseline_bioclip_end_to_end(tmp_path: Path, monkeypatch) -> None:
    taxa = [
        contracts.Taxon(idx=0, species="Ficus religiosa", genus="Ficus", family="Moraceae"),
        contracts.Taxon(idx=1, species="Azadirachta indica", genus="Azadirachta", family="Meliaceae"),
        contracts.Taxon(idx=2, species="Mangifera indica", genus="Mangifera", family="Anacardiaceae"),
    ]
    labels_path = tmp_path / "labels.json"
    contracts.save_taxa(taxa, labels_path)

    samples = []
    for i, label in enumerate([0, 1, 2, 0]):
        img_path = tmp_path / f"img_{i}.png"
        Image.new("RGB", (32, 32), color=(i * 40, 100, 50)).save(img_path)
        samples.append(
            contracts.Sample(
                image_id=f"test/{i}",
                path=str(img_path),
                label=label,
                source="test",
                group=f"g{i}",
            )
        )
    manifest_path = tmp_path / "toy_test.jsonl"
    data.write_manifest(samples, manifest_path)

    class _FakeCustomLabelsClassifier:
        def __init__(self, cls_ary: list[str], device: str = "cpu") -> None:
            self.classes = list(cls_ary)

        def predict(self, images: list[str], k: int | None = None, batch_size: int = 10) -> list[dict]:
            out = []
            # Return deliberately reversed class order per image to verify taxa-order mapping
            probs_by_cls = {
                "Ficus religiosa": [0.7, 0.1, 0.2, 0.8],
                "Azadirachta indica": [0.2, 0.8, 0.1, 0.1],
                "Mangifera indica": [0.1, 0.1, 0.7, 0.1],
            }
            for idx, path in enumerate(images):
                for cls_name in reversed(self.classes):
                    out.append(
                        {
                            "file_name": path,
                            "classification": cls_name,
                            "score": probs_by_cls[cls_name][idx],
                        }
                    )
            return out

    class _FakeTreeOfLifeClassifier:
        def __init__(self, device: str = "cpu") -> None:
            self.device = device

        def predict(self, images: list[str], rank, k: int = 5, batch_size: int = 10) -> list[dict]:
            out = []
            top5_by_img = [
                ["Ficus religiosa", "Ficus benghalensis", "Morus alba", "Rosa canina", "Poa annua"],
                ["Melia azedarach", "Azadirachta indica", "Cedrela odorata", "Swietenia macrophylla", "Toona ciliata"],
                ["Anacardium occidentale", "Pistacia vera", "Rhus typhina", "Toxicodendron radicans", "Schinus molle"],
                ["Ficus religiosa", "Ficus elastica", "Ficus carica", "Ficus lyrata", "Ficus pumila"],
            ]
            for idx, path in enumerate(images):
                for sp in top5_by_img[idx][:k]:
                    out.append({"file_name": path, "species": sp, "score": 0.2})
            return out

    monkeypatch.setattr(baseline_bioclip, "_install_bioclip_memory_patches", lambda: None)
    monkeypatch.setattr(baseline_bioclip, "CustomLabelsClassifier", _FakeCustomLabelsClassifier)
    monkeypatch.setattr(baseline_bioclip, "TreeOfLifeClassifier", _FakeTreeOfLifeClassifier)
    monkeypatch.setattr(baseline_bioclip.harness, "RUNS_DIR", tmp_path / "runs")

    res = baseline_bioclip.run_dataset(
        dataset="toy",
        manifest_path=manifest_path,
        labels_path=labels_path,
        device="cpu",
        batch_size=4,
    )

    assert res["n"] == 4
    assert np.isclose(res["top1"], 1.0)
    assert np.isclose(res["tol_top1"], 0.5)
    assert np.isclose(res["tol_top5"], 0.75)

    saved = json.loads((tmp_path / "runs" / "L2" / "b2_toy.json").read_text(encoding="utf-8"))
    assert saved["top1"] == 1.0
    assert saved["tol_top1"] == 0.5
    assert saved["tol_top5"] == 0.75
