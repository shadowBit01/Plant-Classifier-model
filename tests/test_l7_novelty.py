"""Lane L7 (card L7-e): related work, head-to-head table and novelty section of the report."""
from __future__ import annotations

import copy
import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("build_report_l7e", REPO / "pitch" / "report" / "build_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_novelty_section_and_citations():
    br = _load()
    html = br.build_html().read_text(encoding="utf-8")
    for s in ("Related work and what is new", "What we built on", "What we implemented",
              "How good it is against them", "What is new in VanaLens", "Related-work references"):
        assert s in html, s
    cited = set(re.findall(r"\[R(\d+)", html))
    listed = set(re.findall(r"<strong>R(\d+)</strong>", html))
    assert cited and cited <= listed, cited - listed      # every cited code has a reference entry
    assert "NOT comparable" in html                       # reported accuracies are labelled as such
    assert "—" not in html


def test_h2h_rows_fallback_and_vanalens_filled():
    br = _load()
    rows = br.h2h_rows("vl600")
    assert len(rows) == len(br.H2H)
    ours = [r for r in rows if r["ours"]]
    assert ours and all(r["top1"][0] is not None for r in ours)
    zs = next(r for r in rows if r["label"].startswith("BioCLIP 2 zero-shot, 607"))
    assert zs["top1"][0] is not None


def test_h2h_reads_l6e_baselines_block_in_two_layouts():
    br = _load()
    fin = copy.deepcopy(br.J("L6/final.json"))
    layouts = [
        {"vl600": [{"method": "BioCLIP-only linear probe", "top1": 0.91, "ci": {"top1": [0.90, 0.92]}},
                   {"method": "DINOv2-only linear probe", "top1": {"value": 0.80, "ci": [0.79, 0.81]}}]},
        {"vl600": {"bioclip_only_probe": {"top1": 0.91, "top1_ci": [0.90, 0.92]},
                   "dino_only_probe": {"top1": 0.80}}},
    ]
    for block in layouts:
        fin["baselines"] = block
        br._cache["L6/final.json"] = fin
        rows = {r["label"]: r for r in br.h2h_rows("vl600")}
        bio, dino = rows["BioCLIP 2 features + linear probe"], rows["DINOv2 features + linear probe"]
        assert bio["source"] == "L6-e" and abs(bio["top1"][0] - 0.91) < 1e-9 and bio["top1"][1] == [0.90, 0.92]
        assert dino["source"] == "L6-e" and abs(dino["top1"][0] - 0.80) < 1e-9
        assert br.h2h_derived()["h2h_l6e"] == "yes"
