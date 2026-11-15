"""L6-e: baselines block of scripts/final_table.py on synthetic clusters (no data, CPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import final_table as ft  # noqa: E402


def _data(clusters, aug: bool):
    g = clusters["gallery"]
    Xq, yq = clusters["Xq"], clusters["yq"]
    half = len(yq) // 2
    idx = np.arange(len(yq))
    cal, test = idx % 2 == 0, idx % 2 == 1
    groups = [f"obs{i // 2}" for i in range(int(test.sum()))]       # 2 photos per observation
    boot = ft.Boot(groups, n_boot=50)
    Xa = g.X[::2] if aug else np.empty((0, g.X.shape[1]), np.float32)
    ya = g.y[::2] if aug else np.empty(0, np.int64)
    assert half > 0
    return {"Xg": g.X, "yg": g.y, "ids": g.ids, "Xa": Xa, "ya": ya, "Xc": Xq[cal], "yc": yq[cal],
            "Xt": Xq[test], "yt": yq[test], "gt": boot.gidx(groups)}, boot


def test_compute_baselines_rows(clusters, text_protos):
    data, boot = _data(clusters, aug=True)
    state = {"knn": {"k": 5, "temperature": 0.05, "dino_weight": 0.25},
             "probe": {"config": {"hidden": 0, "balanced": True, "epochs": 15, "patience": 5, "scale": 10.0,
                                  "dino_weight": 0.5, "weight_decay": 0.0}},
             "text": {"temperature": 0.01}}
    main = [{"method": "+ geo prior", "status": "ok", "n": 25, "top1": 0.9, "top5": 1.0, "macro_top1": 0.9,
             "ci": {"top1": [0.8, 1.0], "top5": [1.0, 1.0], "macro_top1": [0.8, 1.0]}, "note": "per photo"}]
    b2 = {"n": 25, "top1": 0.5, "top5": 0.8, "macro_top1": 0.5}
    rows = ft.compute_baselines(data, state, text_protos, 5, boot, b2, main, device="cpu")
    names = [r["method"] for r in rows]
    assert names == ["B2 BioCLIP 2 zero-shot", "BioCLIP 2-only linear probe", "DINO-only linear probe",
                     "kNN only (both blocks)", "Concatenated-feature linear probe (no fusion)",
                     "Text prototypes only (0 images/species)", "VanaLens full, 1 photo + geo"]
    by = {r["method"]: r for r in rows}
    assert by["B2 BioCLIP 2 zero-shot"]["ci"]["top1"] == "not available"
    assert by["B2 BioCLIP 2 zero-shot"]["top1"] == 0.5
    assert by["VanaLens full, 1 photo + geo"]["ci"]["top1"] == [0.8, 1.0]
    for r in rows[1:6]:
        assert r["n"] == len(data["yt"])
        for c in ft.BCOLS:
            lo, hi = r["ci"][c]
            assert 0.0 <= lo <= r[c] <= hi <= 1.0
        assert r["top1"] <= r["top5"]
        assert r["top1"] > 0.6, r                 # separable clusters: every scorer beats chance (0.2) clearly


def test_compute_baselines_without_aug_or_b2(clusters):
    data, boot = _data(clusters, aug=False)
    rows = ft.compute_baselines(data, {"probe": {"config": {"epochs": 5}}}, None, 5, boot, None, [], device="cpu")
    names = [r["method"] for r in rows]
    assert "Text prototypes only (0 images/species)" not in names
    assert rows[0]["top1"] is None and rows[0]["ci"]["top1"] == "not available"
    assert not any(n.startswith("VanaLens") for n in names)
