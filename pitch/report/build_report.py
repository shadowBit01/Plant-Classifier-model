"""Lane L7 (pitch). Build the VanaLens project report: markdown -> styled HTML -> PDF (headless Edge).

Usage: python pitch/report/build_report.py [--no-pdf]

Source:  pitch/report/VanaLens_report.md, with placeholders that are filled from runs/*.json at build time,
         so a re-run of the experiments needs only one rebuild (no hand-copied numbers):
  {{row:DS:METHOD:COL}}     final.json row value, as a percentage        e.g. {{row:vl600:+ geo prior:top1}}
  {{rowci:DS:METHOD:COL}}   same, with its 95% bootstrap CI
  {{pct:FILE:PATH}}         any value in runs/FILE (dotted PATH, list indices allowed), as a percentage;
                            a {"value", "ci"} dict prints value only
  {{pctci:FILE:PATH}}       same, with CI when the value is a {"value", "ci"} dict
  {{num:FILE:PATH:FMT}}     raw number with a Python format spec (e.g. .3f, ,d)
  {{d:NAME}}                derived sentence fragment or number computed in derived() (e.g. geo-prior effect)
  {{table:NAME}}            generated markdown table (with its "Table:" caption line), see TABLES
  {{fig:NAME}}              figure with an auto-numbered caption, see figures()
A value that is missing or null prints "not measured".

Outputs: pitch/report/VanaLens_report.html, pitch/report/VanaLens_report.pdf, pitch/report/figs/*.png,
         pitch/report/numbers.json (every filled placeholder and its value: the generated include, for audit).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUNS = REPO / "runs"
FIGS = HERE / "figs"
PITCH = HERE.parent
DATA = Path(os.environ.get("VANALENS_DATA", Path.home() / "vanalens-data"))
CACHE = Path(os.environ.get("VANALENS_CACHE", Path.home() / "vanalens-cache"))
SRC = HERE / "VanaLens_report.md"
OUT_HTML = HERE / "VanaLens_report.html"
OUT_PDF = HERE / "VanaLens_report.pdf"
NM = "not measured"
DINO_FIX_TIME = "2026-11-15 19:55"   # docs/HANDOFF.md: DINO weight-loading bug found; fixed by L2-f, refit by L3-d

GREEN, GREY, INK, MUTED = "#2e7d32", "#b8bec6", "#1f2933", "#66707a"
PHOTO_SET = {   # same CC-licensed vl600 photos as the deck (scripts/build_deck.py), cropped there to pitch/photos/
    "calotropis": ("Calotropis gigantea/292293824_525990526.jpg", "Calotropis gigantea"),
    "delonix": ("Delonix regia/288938584_519658590.jpg", "Delonix regia"),
    "hibiscus": ("Hibiscus rosa-sinensis/183307220_319780364.jpg", "Hibiscus rosa-sinensis"),
    "lantana": ("Lantana camara/288661222_519141340.jpg", "Lantana camara"),
}


# ------------------------------------------------------------------ data access
_cache: dict[str, Any] = {}


def J(rel: str) -> Any:
    """runs/<rel> as JSON, or None when the file is missing."""
    if rel not in _cache:
        p = RUNS / rel
        _cache[rel] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return _cache[rel]


def dig(obj: Any, path: str) -> Any:
    """Follow a dotted path; integer parts index lists. None if anything is missing."""
    for part in [p for p in path.split(".") if p]:
        if obj is None:
            return None
        if isinstance(obj, list):
            try:
                obj = obj[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


def frow(ds: str, method: str) -> dict | None:
    """The final.json row for (dataset, method), or None."""
    fin = J("L6/final.json") or {}
    for r in dig(fin, f"datasets.{ds}") or []:
        if r.get("method") == method:
            return r
    return None


def rv(ds: str, method: str, col: str) -> float | None:
    r = frow(ds, method)
    return None if r is None or r.get("status") != "ok" else r.get(col)


def rci(ds: str, method: str, col: str) -> list | None:
    r = frow(ds, method)
    return None if r is None else (r.get("ci") or {}).get(col)


def pct(x: Any, nd: int = 1) -> str:
    if isinstance(x, dict):
        x = x.get("value")
    return NM if x is None else f"{100 * float(x):.{nd}f}%"


def pct_ci(x: Any, ci: list | None = None, nd: int = 1) -> str:
    if isinstance(x, dict):
        x, ci = x.get("value"), x.get("ci")
    if x is None:
        return NM
    s = f"{100 * float(x):.{nd}f}%"
    if ci:
        s += f" (95% CI {100 * ci[0]:.{nd}f} to {100 * ci[1]:.{nd}f})"
    return s


def cell(x: Any, ci: list | None = None) -> str:
    """Table cell: value with a compact CI on a second line."""
    if isinstance(x, dict):
        x, ci = x.get("value"), x.get("ci")
    if x is None:
        return "n/m"
    s = f"{100 * float(x):.1f}"
    if ci:
        s += f" <span class='ci'>[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]</span>"
    return s


def val(x: Any) -> float | None:
    return x.get("value") if isinstance(x, dict) else x


# ------------------------------------------------------------------ derived text
def points(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else 100 * (b - a)


def effect(a: float | None, b: float | None, what: str = "top-1") -> str:
    """'raised top-1 by 0.6 points' / 'lowered ...' / 'did not change ...'."""
    d = points(a, b)
    if d is None:
        return f"had an effect on {what} that was {NM}"
    if abs(d) < 0.05:
        return f"did not change {what}"
    return f"{'raised' if d > 0 else 'lowered'} {what} by {abs(d):.1f} points"


def reaches(ds: str) -> tuple[str, str]:
    tg = dig(J("L6/final.json"), f"extras.{ds}.target") or []
    yes = [t["metric"] for t in tg if t.get("reaches_95")]
    no = [t["metric"] for t in tg if not t.get("reaches_95")]
    return (", ".join(yes) or "none"), (", ".join(no) or "none")


def post_fix() -> bool:
    created = dig(J("L3/ablation_vl600.json"), "state.created") or ""
    return created > DINO_FIX_TIME


def manifest_counts(ds: str) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for sp in ("gallery", "cal", "test"):
        p = CACHE / "manifests" / f"{ds}_{sp}.jsonl"
        out[sp] = sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else None
    return out


def licences(ds: str) -> str:
    p = DATA / ds / "meta.csv"
    if not p.exists():
        return NM
    counts: dict[str, int] = {}
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r.get("license") or "?").upper().replace("CC-", "CC ")
            counts[k] = counts.get(k, 0) + 1
    tot = sum(counts.values())
    return ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


def labels_count(ds: str) -> int | None:
    p = DATA / ds / "labels.json"
    return len(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None


def wilson(p: Any, n: Any) -> str:
    """95% Wilson score interval for a proportion p over n trials, as '95% CI a to b' (percent)."""
    try:
        p, n = float(p), int(n)
    except (TypeError, ValueError):
        return NM
    if n <= 0:
        return NM
    z = 1.96
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return f"95% CI {100 * (mid - half):.1f} to {100 * (mid + half):.1f}"


def dino_note(short: bool = False) -> str:
    """What the DINO block adds, from the L6-e single-encoder probes (BioCLIP 2-only vs DINO-only vs both)."""
    ents = {e["name"].lower(): e for e in _bl_entries("vl600")}
    def find(*keys):
        for k, e in ents.items():
            if all(x in k for x in keys):
                return _metric(e, "top1")
        return (None, None)
    bc, dn, both = find("bioclip", "only"), find("dino", "only"), find("concatenated")
    if bc[0] is None or dn[0] is None or both[0] is None:
        return ("the kNN row is the cleanest test of whether the second encoder adds information, since kNN uses "
                "the whole vector without any training.")
    gain = 100 * (both[0] - bc[0])
    overlap = bool(bc[1] and both[1] and bc[1][1] >= both[1][0])
    tail = ("the two intervals overlap" if overlap else "the two intervals do not overlap")
    if short:
        return (f"a linear probe on BioCLIP 2 features alone reaches {pct(bc[0])} top-1 against {pct(both[0])} on "
                f"both encoders ({gain:+.1f} points, {tail})")
    return (f"The baselines in Table H2H measure what the second encoder adds: a linear probe on BioCLIP 2 "
            f"features alone reaches {pct_ci(*bc)}, on DINOv2 features alone {pct_ci(*dn)}, and on both "
            f"{pct_ci(*both)}. DINOv2 is clearly weaker on its own and adds {gain:+.1f} points to BioCLIP 2 "
            f"({tail}); most of the visual signal comes from the biology-trained encoder.")


def train_rows(ds: str) -> str:
    """Rows the heads train on: gallery photos plus their stored augmented views (read from the .npz headers only)."""
    import zipfile
    import numpy as np
    p = CACHE / "emb" / f"{ds}_gallery.npz"
    if not p.exists():
        return NM
    tot = 0
    try:
        with zipfile.ZipFile(p) as z:
            for name in ("X.npy", "X_aug.npy"):
                if name not in z.namelist():
                    continue
                with z.open(name) as f:
                    ver = np.lib.format.read_magic(f)
                    fn = (np.lib.format.read_array_header_1_0 if ver == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                    shape = fn(f)[0]
                tot += int(shape[0])
    except Exception:
        return NM
    return f"{tot:,}" if tot else NM


def dino_gain() -> str:
    """Top-1 points the DINOv2 block adds to a BioCLIP 2-only linear probe (L6-e baselines), e.g. '+0.2 points'."""
    ents = {e["name"].lower(): e for e in _bl_entries("vl600")}
    bc = next((_metric(e, "top1")[0] for k, e in ents.items() if "bioclip" in k and "only" in k), None)
    both = next((_metric(e, "top1")[0] for k, e in ents.items() if "concatenated" in k), None)
    return NM if bc is None or both is None else f"{100 * (both - bc):+.1f} points"


def derived() -> dict[str, str]:
    d: dict[str, str] = {}
    for ds, tag in (("vl600", "vl"), ("inat_india", "in")):
        one = rv(ds, "VanaLens 1 photo (no geo)", "top1")
        geo = rv(ds, "+ geo prior", "top1")
        multi = rv(ds, "VanaLens up to 3 photos + geo", "top1")
        b2 = rv(ds, "B2 BioCLIP 2 kNN", "top1")
        d[f"geo_effect_{tag}"] = effect(one, geo)
        d[f"multi_effect_{tag}"] = effect(geo, multi)
        g = points(b2, geo)
        d[f"gain_b2_{tag}"] = NM if g is None else f"{abs(g):.1f} points {'higher' if g >= 0 else 'lower'}"
        note = (frow(ds, "VanaLens up to 3 photos + geo") or {}).get("note", "")
        mm = re.search(r"(\d+)/(\d+) observations have 2-3 photos", note)
        d[f"multi_obs_{tag}"] = NM if not mm else f"{int(mm.group(1)):,} of {int(mm.group(2)):,}"
        y, n = reaches(ds)
        d[f"reach_yes_{tag}"], d[f"reach_no_{tag}"] = y, n
        ex = dig(J("L6/final.json"), f"extras.{ds}") or {}
        d[f"beta_{tag}"] = NM if ex.get("beta") is None else f"{ex['beta']:g}"
        d[f"n_obs_{tag}"] = NM if ex.get("n_test_observations") is None else f"{ex['n_test_observations']:,}"
        d[f"n_photos_{tag}"] = NM if ex.get("n_test_photos") is None else f"{ex['n_test_photos']:,}"
        d[f"lic_{tag}"] = licences(ds)
        n = labels_count(ds)
        d[f"classes_{tag}"] = NM if n is None else str(n)
        for sp, c in manifest_counts(ds).items():
            d[f"{sp}_{tag}"] = NM if c is None else f"{c:,}"
        e2e = J(f"L6/e2e_{ds}.json") or {}
        d[f"e2e_n_{tag}"] = str(e2e.get("n", NM))
        d[f"e2e_top1_{tag}"] = pct(e2e.get("top1"))
        d[f"e2e_unknown_{tag}"] = pct(e2e.get("unknown_rate"))
        w = e2e.get("seconds_per_obs_warm")
        d[f"e2e_sec_{tag}"] = NM if w is None else f"{w:.1f} s"
        d[f"e2e_device_{tag}"] = str(e2e.get("device", NM)).upper()
    ab = J("L3/ablation_vl600.json") or {}
    d["state_created"] = dig(ab, "state.created") or NM
    d["fix_status"] = (
        "The numbers in this report were recomputed after that fix (cards L2-f, L3-d and L6-d), so the DINO half of "
        "every embedding now holds real DINOv2-large features."
        if post_fix() else
        "The numbers in this version of the report were computed BEFORE that fix: the DINO half of every cached "
        "embedding was effectively random, so the results below are close to BioCLIP 2 features alone. The fix and a "
        "full re-run (cards L2-f, L3-d, L6-d) were in progress when this version was built; rebuilding this report "
        "after them refreshes every number and table."
    )
    d["knn_note"] = (
        "The kNN scorer is the weakest here, consistent with the broken DINO half described in Section 4.11: kNN is "
        "the only scorer that cannot learn to ignore part of the vector." if not post_fix() else
        dino_note())
    fin = J("L6/final.json")
    e2e_unk = (J("L6/e2e_vl600.json") or {}).get("unknown_rate")
    d["e2e_note"] = (
        f"{d['e2e_unknown_vl']} of the answers were flagged \"not in catalogue\". "
        + ("Before the DINO fix, live and cached DINO vectors disagree, so every live photo looks far from the gallery; "
           "this is the clearest symptom of the bug in Section 4.11 and the first number to recheck after the fix."
           if not post_fix() else
           "This run used the same weight-verified encoders as the cache (Section 4.13), so live and cached vectors "
           "agree; these flags come from the open-set gate itself (Section 7.6).")
        if e2e_unk is not None else "The share flagged \"not in catalogue\" was not measured.")
    sp, fam = rv("vl600", "+ geo prior", "top1"), rv("vl600", "+ geo prior", "family_top1")
    d["fam_recover_vl"] = NM if sp is None or fam is None or sp >= 1 else f"{100 * (fam - sp) / (1 - sp):.0f}%"
    tg = dig(fin, "extras.inat_india.target") or []
    d["ci_below_in"] = f"{sum(1 for t in tg if not t.get('ci_low_reaches_95'))} of {len(tg)}" if tg else NM
    ms = rv("vl600", "+ geo prior", "mean_set_size")
    d["sets_note"] = (
        "and they are almost always a single species: with a model that is already more than 90% accurate, the "
        "threshold rule rarely needs a second species to reach 90% coverage, so the sets carry little extra "
        "information on this test set (Section 8 gives the next step)"
        if ms is not None and ms < 1.05 else
        f"with a mean set size of {NM if ms is None else f'{ms:.2f}'} species on vl600")
    sel = val(dig(fin, "extras.vl600.selective.test_accuracy_answered"))
    share = val(dig(fin, "extras.vl600.selective.test_share_answered"))
    cal_acc = val(dig(fin, "extras.vl600.selective.cal_accuracy_answered"))
    if sel is not None and sel < 0.95 and share is not None and share >= 0.999:
        d["selective_note"] = (
            f"below the target, and with {pct(share)} of test photos answered the selective step does nothing on "
            f"vl600: calibration accuracy was already {pct(cal_acc)} with every photo answered, so the threshold "
            "chosen there rejects nothing, while the post-cutoff test photos are harder")
    else:
        d["selective_note"] = (
            "below the target: the post-cutoff test photos are harder than the pre-cutoff calibration photos, so a "
            "threshold tuned on calibration is too permissive on test"
            if sel is not None and sel < 0.95 else "which meets the target on test as well")
    qh = dig(ab, "state.ensemble.qhat") or dig(ab, "state.qhat")
    d["qhat_vl"] = NM if qh is None else f"{qh:.3f}"
    d["dino_benefit"] = (
        "and the system has not yet been shown to benefit from a second encoder" if not post_fix() else
        "and even after the fix the second encoder adds little: " + dino_note(short=True))
    e_f, e_t = dig(ab, "rows.fused.ece"), dig(ab, "rows.fused+T.ece")
    if e_f is None or e_t is None:
        d["temp_note"] = "the temperature step leaves top-1 unchanged (its effect on calibration error was not measured)"
    elif e_t < e_f - 0.001:
        d["temp_note"] = (f"the temperature step cuts calibration error from {pct(e_f)} to {pct(e_t)} "
                          "without changing top-1")
    else:
        d["temp_note"] = (f"the temperature step changes neither top-1 nor calibration error here ({pct(e_f)} before, "
                          f"{pct(e_t)} after), because the fused scores were already well calibrated on this test set")
    e2e = J("L6/e2e_vl600.json") or {}
    d["e2e_ci_vl"] = wilson(e2e.get("top1"), e2e.get("n"))
    dp = Path(__file__).resolve().parents[1] / "demo_picks.json"     # written by pitch/demo_pick.py (L7-d)
    u = (json.loads(dp.read_text(encoding="utf-8")) if dp.exists() else {}).get("unknown") or {}
    if u.get("n_test"):
        conf_wrong = sum(1 for t in u["test"] if t["before"]["rank"] == "species" and t["before"]["conf"] >= 0.7)
        d["unknown_case"] = (
            f"A live-style check makes the same point. {u['species']}, an orchid in inat_india that is not among the "
            f"607 vl600 species, was shown to the vl600 model: the gate rejected {u['rejected_before']} of its "
            f"{u['n_test']} test photos, and {conf_wrong} were named as a related orchid species with at least 70% "
            f"confidence. After appending {u['shots']} of its photos (no retraining), {u['named_after']} of the "
            f"{u['n_test']} were named correctly. The gate does not yet guard against close relatives of known "
            "species, but adding a few photos resolves such a case at once (cached embeddings, `pitch/demo_pick.py`).")
    else:
        d["unknown_case"] = ""
    rs_path = Path(__file__).with_name("review_stats.json")      # written by review_stats.py (L7-d)
    rs = json.loads(rs_path.read_text(encoding="utf-8")) if rs_path.exists() else {}
    d["india_test_vl"] = NM if rs.get("india_share_test_observations") is None else pct(rs["india_share_test_observations"])
    tob, tph = rs.get("test_obs_per_species") or {}, rs.get("test_photos_per_species") or {}
    d["test_obs_sp_vl"] = NM if not tob else f"{tob['min']} to {tob['max']} (median {tob['median']:g})"
    d["test_ph_sp_vl"] = NM if not tph else f"{tph['min']} to {tph['max']} (median {tph['median']:g})"
    d["same_spot_vl"] = (NM if rs.get("test_obs_same_spot_as_gallery_or_cal") is None else
                         f"{rs['test_obs_same_spot_as_gallery_or_cal']:,} of {rs['test_obs_with_gps']:,} "
                         f"({pct(rs['same_spot_share'])})")
    d["fix_short"] = "after the DINO loader fix" if post_fix() else "before the DINO loader fix, so the DINO half is random"
    d["train_rows_vl"] = train_rows("vl600")
    d["dino_gain_short"] = dino_gain()
    ens = dig(ab, "state.ensemble.weights") or {}
    d["ens_weights"] = ", ".join(f"{k} {v:g}" for k, v in ens.items()) or NM
    fw = dig(ab, "state.weights") or {}
    d["fused_weights"] = ", ".join(f"{k} {v:g}" for k, v in fw.items()) or NM
    d.update(h2h_derived())
    return d


# ------------------------------------------------------------------ tables
def t_main(ds: str) -> str:
    b2 = J(f"L2/b2_{ds}.json") or {}
    name = "vl600" if ds == "vl600" else "inat_india"
    rows = [
        ("B0 Pl@ntNet API", "B0 PlantNet API", "not run: no API key"),
        ("B2 BioCLIP 2 zero-shot, open-ended (tree of life)", None, "point value, no CI"),
        ("B2 BioCLIP 2 zero-shot, closed set (all class names)", "B2 BioCLIP 2 kNN", "point value, no CI"),
        ("B3 linear probe on our embeddings", "B3 linear probe", ""),
        ("B4 general VLM, open-ended", "B4 VLM", "not run: no LLM key"),
        ("VanaLens, 1 photo, no location", "VanaLens 1 photo (no geo)", ""),
        ("VanaLens, 1 photo + location prior", "+ geo prior", ""),
        ("VanaLens, up to 3 photos + location", "VanaLens up to 3 photos + geo", "per observation"),
        ("VanaLens + VLM reranker", "+ reranker", "not run: no LLM key"),
    ]
    out = [f"Table: Test results on {name} ({{{{d:n_photos_{'vl' if ds == 'vl600' else 'in'}}}}} photos). "
           "Accuracy in %, 95% bootstrap CI over observations in brackets; n/m = not measured.",
           "", "| Method | Top-1 | Top-5 | Macro top-1 | Genus top-1 | Family top-1 | Note |",
           "|---|---|---|---|---|---|---|"]
    for label, method, note in rows:
        if method is None:
            t1, t5 = b2.get("tol_top1"), b2.get("tol_top5")
            out.append(f"| {label} | {cell(t1)} | {cell(t5)} | n/m | n/m | n/m | {note} |")
            continue
        r = frow(ds, method)
        if r is None or r.get("status") != "ok":
            out.append(f"| {label} | n/m | n/m | n/m | n/m | n/m | {note or (r or {}).get('note', 'not run')} |")
            continue
        ci = r.get("ci") or {}
        cells = [cell(r.get(c), ci.get(c)) for c in ("top1", "top5", "macro_top1", "genus_top1", "family_top1")]
        bold = label.startswith("VanaLens, ")
        if bold:
            label = f"**{label}**"
        out.append(f"| {label} | " + " | ".join(cells) + f" | {note} |")
    return "\n".join(out)


def t_ablation(ds: str) -> str:
    ab = J(f"L3/ablation_{ds}.json") or {}
    rows = ab.get("rows") or {}
    desc = {
        "knn": "kNN over the reference gallery", "probe": "linear probe", "mlp": "MLP head (1 hidden layer)",
        "text": "text-trait prototypes (no training)", "fused": "log-linear fusion of knn, probe, text",
        "fused+T": "fused + temperature scaling", "fused+T+sets": "fused + T + 90% conformal sets",
        "ensemble": "final: probe + MLP + text (+ knn) fused, + T, + sets",
    }
    name = "vl600" if ds == "vl600" else "inat_india"
    out = [f"Table: Ablation on {name} test (single photo, no location). Top-1 with 95% CI; NLL and ECE "
           "(expected calibration error) show calibration; cal top-1 is the accuracy on the calibration split that "
           "chose the configuration.", "",
           "| Component | Top-1 | Top-5 | NLL | ECE | Cal top-1 |", "|---|---|---|---|---|---|"]
    for k, lab in desc.items():
        r = rows.get(k)
        if r is None:
            out.append(f"| {lab} | n/m | n/m | n/m | n/m | n/m |")
            continue
        t1 = cell(r.get("top1"), rci(ds, k, "top1"))
        nll = f"{r['nll']:.3f}" if r.get("nll") is not None else "n/m"
        ece = f"{100 * r['ece']:.1f}" if r.get("ece") is not None else "n/m"
        lab = f"**{lab}**" if k == ab.get("final_row") else lab
        out.append(f"| {lab} | {t1} | {cell(r.get('top5'))} | {nll} | {ece} | {cell(r.get('cal_top1'))} |")
    return "\n".join(out)


def t_uncertainty() -> str:
    out = ["Table: Uncertainty outputs on the test splits (VanaLens, 1 photo + location unless noted). "
           "Backoff accuracy = share of answers correct at the rank reported (species, genus or family); "
           "species share = share of answers given at species rank.", "",
           "| Quantity | vl600 | inat_india |", "|---|---|---|"]

    def both(f):
        return " | ".join(f(ds) for ds in ("vl600", "inat_india"))

    fin = J("L6/final.json")
    out.append("| Conformal set coverage (target 90%) | " + both(lambda ds: cell(rv(ds, "+ geo prior", "coverage"), rci(ds, "+ geo prior", "coverage"))) + " |")
    out.append("| Mean conformal set size | " + both(lambda ds: "n/m" if rv(ds, "+ geo prior", "mean_set_size") is None else f"{rv(ds, '+ geo prior', 'mean_set_size'):.3f}") + " |")
    out.append("| Backoff accuracy at reported rank | " + both(lambda ds: cell(rv(ds, "+ geo prior", "backoff_acc"), rci(ds, "+ geo prior", "backoff_acc"))) + " |")
    out.append("| Answers given at species rank | " + both(lambda ds: cell(rv(ds, "+ geo prior", "species_share"), rci(ds, "+ geo prior", "species_share"))) + " |")
    out.append("| Selective: threshold on confidence (fit on cal) | " + both(lambda ds: "n/m" if dig(fin, f"extras.{ds}.selective.threshold") is None else f"{dig(fin, f'extras.{ds}.selective.threshold'):.3f}") + " |")
    out.append("| Selective: accuracy on cal when answered | " + both(lambda ds: cell(dig(fin, f"extras.{ds}.selective.cal_accuracy_answered"))) + " |")
    out.append("| Selective: accuracy on test when answered | " + both(lambda ds: cell(dig(fin, f"extras.{ds}.selective.test_accuracy_answered"))) + " |")
    out.append("| Selective: share of test photos answered | " + both(lambda ds: cell(dig(fin, f"extras.{ds}.selective.test_share_answered"))) + " |")
    out.append("| Calibration error ECE, final head (ablation) | " + both(lambda ds: "n/m" if dig(J(f"L3/ablation_{ds}.json"), "rows.ensemble.ece") is None else f"{100 * dig(J(f'L3/ablation_{ds}.json'), 'rows.ensemble.ece'):.1f}") + " |")
    return "\n".join(out)


def t_openset() -> str:
    fin = J("L6/final.json")
    out = ["Table: Open-set test: a random 10% of species is removed from the model (seed 0), the rest is refit on "
           "cal, and the open-set gate (best gallery cosine below a threshold) is scored. Values in %, CI in brackets.",
           "", "| Quantity | vl600 | inat_india |", "|---|---|---|"]

    def g(path, fmt=None):
        vals = []
        for ds in ("vl600", "inat_india"):
            x = dig(fin, f"extras.{ds}.open_set.{path}")
            vals.append(("n/m" if x is None else format(x, fmt)) if fmt else cell(x))
        return " | ".join(vals)

    out.append("| Species held out / kept | " + " | ".join(
        f"{dig(fin, f'extras.{ds}.open_set.held_out_species')} / {dig(fin, f'extras.{ds}.open_set.known_species')}"
        for ds in ("vl600", "inat_india")) + " |")
    out.append("| Test photos of held-out species | " + g("n_test_held_out", ",d") + " |")
    out.append("| Held-out photos rejected (want high) | " + g("reject_rate_held_out") + " |")
    out.append("| Known photos wrongly rejected (want low) | " + g("false_reject_rate_known") + " |")
    out.append("| AUROC of the gate score | " + g("auroc") + " |")
    out.append("| Known-species top-1 after refit | " + g("known_top1_refit") + " |")
    return "\n".join(out)


def t_fewshot() -> str:
    fin = J("L6/final.json")
    out = ["Table: Adding species without retraining. The held-out species of Table OPENSET are added back with "
           "1, 3, 5 or 10 gallery photos each (add_species: append to the kNN gallery, add a text prototype). "
           "Top-1 over all classes on their test photos, and top-1 on the known species afterwards. %, CI in brackets.",
           "", "| Shots per new species | vl600: new species | vl600: known after | inat_india: new species | inat_india: known after |",
           "|---|---|---|---|---|"]
    fv = dig(fin, "extras.vl600.few_shot") or []
    fi = dig(fin, "extras.inat_india.few_shot") or []
    shots = sorted({r["shots"] for r in fv + fi})
    for s in shots:
        a = next((r for r in fv if r["shots"] == s), {})
        b = next((r for r in fi if r["shots"] == s), {})
        out.append(f"| {s} | {cell(a.get('top1_new'))} | {cell(a.get('top1_known_after'))} | "
                   f"{cell(b.get('top1_new'))} | {cell(b.get('top1_known_after'))} |")
    if not shots:
        out.append("| not measured | | | | |")
    return "\n".join(out)


def t_target() -> str:
    fin = J("L6/final.json")
    out = ["Table: Which headline numbers reach the 95% target. 'CI low' = the lower end of the 95% CI also "
           "reaches 95%.", "", "| Metric | vl600 | reaches 95% | CI low | inat_india | reaches 95% | CI low |",
           "|---|---|---|---|---|---|---|"]
    tv = {t["metric"]: t for t in dig(fin, "extras.vl600.target") or []}
    ti = {t["metric"]: t for t in dig(fin, "extras.inat_india.target") or []}
    yn = {True: "yes", False: "no", None: "n/m"}
    for m in list(tv) + [k for k in ti if k not in tv]:
        a, b = tv.get(m, {}), ti.get(m, {})
        out.append(f"| {m} | {cell(a.get('value'), a.get('ci'))} | {yn[a.get('reaches_95')]} | "
                   f"{yn[a.get('ci_low_reaches_95')]} | {cell(b.get('value'), b.get('ci'))} | "
                   f"{yn[b.get('reaches_95')]} | {yn[b.get('ci_low_reaches_95')]} |")
    return "\n".join(out)


def t_dataset() -> str:
    st = J("L1/vl600_stats.json") or {}
    d = derived()
    sp = st.get("species_by_source") or {}
    ps, ob = st.get("photos_per_split") or {}, st.get("observations_per_split") or {}

    def n(x):
        return NM if x is None else f"{x:,}"

    gb = NM if st.get("gb") is None else "%.1f GB" % st["gb"]
    cov = NM if st.get("coverage_pct") is None else "%.1f%%" % st["coverage_pct"]

    out = ["Table: The two evaluation datasets. Splits are by observation: all photos of one iNaturalist observation "
           "fall in the same split.", "",
           "| | vl600 (main) | inat_india |", "|---|---|---|",
           f"| Species | {n(st.get('species'))} ({n(sp.get('india'))} India, {n(sp.get('world'))} world, "
           f"{n(sp.get('common'))} common-Indian extras) | {d['classes_in']} (Madhya Pradesh) |",
           f"| Gallery photos (reference + training) | {n(ps.get('gallery'))} ({n(ob.get('gallery'))} obs.) | {d['gallery_in']} |",
           f"| Calibration photos | {n(ps.get('cal'))} ({n(ob.get('cal'))} obs.) | {d['cal_in']} |",
           f"| Test photos (observed on or after 1 Jul 2025) | {n(ps.get('test'))} ({n(ob.get('test'))} obs.) | "
           f"{d['test_in']} ({d['n_obs_in']} obs.) |",
           f"| Photos per observation | up to 3 | 1 |",
           f"| Size on disk | {gb} | {NM} |",
           f"| Photo licences | {d['lic_vl']} | {d['lic_in']} |",
           f"| Share of India's research-grade plant observations covered by the species list | {cov} | {NM} |"]
    return "\n".join(out)


# ------------------------------------------------------------------ related work + head-to-head (L7-e)
# Head-to-head rows on OUR vl600 test split. Source order: the "baselines" block that L6-e adds to
# runs/L6/final.json (matched by keywords, schema-tolerant), else an existing row of final.json, else n/m.
# (label, keywords that identify the L6-e entry, fallback final.json method, is_vanalens)
H2H = [
    ("BioCLIP 2 zero-shot, open-ended (tree of life)", ("open-ended", "open_ended", "tree of life", "tol"), "@tol", False),
    ("BioCLIP 2 zero-shot, 607 names", ("zero-shot", "zero_shot", "zeroshot", "b2"), "B2 BioCLIP 2 kNN", False),
    ("Text prototypes only (0 images per species)", ("text",), "text", False),
    ("BioCLIP 2 features + linear probe", ("bioclip-only", "bioclip only", "bioclip_only", "bioclip probe",
                                           "bioclip_probe", "bioclip"), None, False),
    ("DINOv2 features + linear probe", ("dinov2-only", "dino-only", "dino only", "dino_only", "dino probe",
                                        "dino_probe", "dino"), None, False),
    ("kNN on both features", ("knn",), "knn", False),
    ("Concatenated features + linear probe", ("concat", "both blocks", "probe"), "probe", False),
    ("VanaLens, 1 photo, no location", ("no geo",), "VanaLens 1 photo (no geo)", True),
    ("VanaLens, 1 photo + location", ("1 photo", "1photo", "one photo", "single"), "+ geo prior", True),
    ("VanaLens, up to 3 photos + location", ("3 photo", "3photo", "three photo", "multi"),
     "VanaLens up to 3 photos + geo", True),
]


def _bl_entries(ds: str) -> list[dict]:
    """L6-e baselines for one dataset as a list of dicts with a 'name' key, whatever the block's exact layout."""
    b = (J("L6/final.json") or {}).get("baselines")
    if b is None:
        return []
    if isinstance(b, dict) and ds in b:
        b = b[ds]                      # {dataset: [...] or {name: {...}}}
    elif isinstance(b, dict) and any(k in b for k in ("vl600", "inat_india")):
        return []                      # keyed by dataset, but not this one
    out = []
    if isinstance(b, dict):
        b = [dict(v, name=k) if isinstance(v, dict) else {"name": k, "top1": v} for k, v in b.items()]
    for e in b if isinstance(b, list) else []:
        if not isinstance(e, dict) or e.get("dataset", ds) != ds:
            continue
        e = dict(e)
        e["name"] = str(e.get("name") or e.get("method") or e.get("baseline") or e.get("id") or "")
        out.append(e)
    return out


def _bl_find(ds: str, keys: tuple[str, ...], used: set[int]) -> dict | None:
    """First unused L6-e entry whose name contains a keyword, trying keywords in order (most specific first)."""
    ents = _bl_entries(ds)
    for k in keys:
        for i, e in enumerate(ents):
            if i not in used and k in e["name"].lower():
                used.add(i)
                return e
    return None


def _metric(e: dict, col: str) -> tuple[float | None, list | None]:
    """(value, ci) for col from an entry that stores either {col: v, ci: {col: [lo, hi]}} or {col: {value, ci}}."""
    v = e.get(col)
    if isinstance(v, dict):
        return v.get("value"), v.get("ci") if isinstance(v.get("ci"), list) else None
    ci = (e.get("ci") or {}).get(col) if isinstance(e.get("ci"), dict) else None
    ci = ci or e.get(f"{col}_ci")
    return v, ci if isinstance(ci, list) and len(ci) == 2 else None


def h2h_rows(ds: str = "vl600") -> list[dict]:
    """Head-to-head rows: label, top1/top5/macro (value, ci), source, is_vanalens."""
    used: set[int] = set()
    rows = []
    b2 = J(f"L2/b2_{ds}.json") or {}
    for label, keys, fb, ours in H2H:
        e = None if ours else _bl_find(ds, keys, used)   # VanaLens rows: always the main table (L6-e copies them)
        r = {"label": label, "ours": ours}
        if e is not None:
            for c in ("top1", "top5", "macro_top1"):
                r[c] = _metric(e, c)
            r["source"] = "L6-e"
        elif fb == "@tol":
            r.update(top1=(b2.get("tol_top1"), None), top5=(b2.get("tol_top5"), None), macro_top1=(None, None),
                     source="runs/L2/b2")
        elif fb is not None and frow(ds, fb) is not None and frow(ds, fb).get("status") == "ok":
            for c in ("top1", "top5", "macro_top1"):
                r[c] = (rv(ds, fb, c), rci(ds, fb, c))
            r["source"] = "runs/L2/b2" if fb.startswith("B2") else "final.json"
        else:
            r.update(top1=(None, None), top5=(None, None), macro_top1=(None, None), source="pending L6-e")
        rows.append(r)
    return rows


def h2h_derived() -> dict[str, str]:
    """Gain of VanaLens (1 photo + location) over BioCLIP 2 zero-shot and over the best baseline row."""
    rows = h2h_rows()
    ours = next(r for r in rows if r["label"].startswith("VanaLens, 1 photo, no"))   # same inputs as baselines
    ours_geo = next(r for r in rows if r["label"].startswith("VanaLens, 1 photo +"))
    ours3 = next(r for r in rows if r["label"].startswith("VanaLens, up to 3"))
    zs = next(r for r in rows if r["label"].startswith("BioCLIP 2 zero-shot, 607"))
    base = [r for r in rows if not r["ours"] and r["top1"][0] is not None]
    best = max(base, key=lambda r: r["top1"][0]) if base else None
    d: dict[str, str] = {}

    def gain(a, b) -> str:
        p = points(a["top1"][0], b["top1"][0])
        return NM if p is None else f"{p:+.1f} points"

    d["h2h_gain_zs"] = gain(zs, ours)
    d["h2h_gain_zs3"] = gain(zs, ours3)
    d["h2h_gain_zs_geo"] = gain(zs, ours_geo)
    d["h2h_gain_best_geo"] = gain(best, ours_geo) if best else NM
    d["h2h_best"] = best["label"].lower() if best else NM
    d["h2h_best_top1"] = pct_ci(*best["top1"]) if best else NM
    d["h2h_gain_best"] = gain(best, ours) if best else NM
    d["h2h_gain_best3"] = gain(best, ours3) if best else NM
    sep = NM
    if best and best["top1"][1] and ours["top1"][1]:
        sep = ("the two 95% intervals do not overlap" if best["top1"][1][1] < ours["top1"][1][0]
               else "the two 95% intervals overlap, so this single-photo gain alone is not statistically clear")
    elif best:
        sep = "the baseline has no interval, so only the VanaLens interval is shown"
    d["h2h_sep"] = sep
    pend = [r["label"] for r in rows if r["source"] == "pending L6-e"]
    d["h2h_pending"] = ("Rows marked n/m (" + "; ".join(pend) + ") are filled by the baselines run (card L6-e) "
                        "and appear on the next rebuild." if pend else
                        "All rows are measured.")
    d["h2h_l6e"] = "yes" if any(r["source"] == "L6-e" for r in rows) else "no"
    return d


def t_h2h() -> str:
    out = ["Table: Head-to-head on the same vl600 test split (12,831 photos, observed on or after 1 July 2025). "
           "Every row uses the same frozen embeddings, gallery and calibration photos; nothing is tuned on test. "
           "Single photo unless noted. Top-1, top-5 and macro top-1 in %, 95% bootstrap CI over observations in "
           "brackets (BioCLIP 2 zero-shot: point value, no per-photo predictions stored).", "",
           "| Method | Images per species used | Top-1 | Top-5 | Macro top-1 | Source |", "|---|---|---|---|---|---|"]
    imgs = {"BioCLIP 2 zero-shot, open-ended (tree of life)": "0 (whole tree of life)",
            "BioCLIP 2 zero-shot, 607 names": "0 (names only)",
            "Text prototypes only (0 images per species)": "0 (names + traits)"}
    src = {"L6-e": "baselines run", "final.json": "main run", "runs/L2/b2": "B2 run", "pending L6-e": "pending"}
    for r in h2h_rows():
        lab = f"**{r['label']}**" if r["ours"] else r["label"]
        cells = " | ".join(cell(*r[c]) for c in ("top1", "top5", "macro_top1"))
        out.append(f"| {lab} | {imgs.get(r['label'], 'gallery photos')} | {cells} | {src[r['source']]} |")
    return "\n".join(out)


def t_caps() -> str:
    """Capabilities the baselines do not have, measured on vl600 test."""
    fin = J("L6/final.json")
    ab = (J("L3/ablation_vl600.json") or {}).get("rows") or {}
    b2 = J("L2/b2_vl600.json") or {}
    fs = {r["shots"]: r for r in dig(fin, "extras.vl600.few_shot") or []}
    ss = rv("vl600", "+ geo prior", "mean_set_size")
    out = ["Table: Capabilities beyond top-1, vl600 test. A dash means the method has no such output.", "",
           "| Capability | BioCLIP 2 zero-shot | Linear probe (concat) | kNN | VanaLens |", "|---|---|---|---|---|",
           f"| Calibration error ECE (lower is better) | {cell(b2.get('ece'))} | {cell(dig(ab, 'probe.ece'))} | "
           f"{cell(dig(ab, 'knn.ece'))} | {cell(dig(ab, 'ensemble.ece'))} |",
           f"| 90% conformal set: coverage / mean size | - | - | - | {cell(rv('vl600', '+ geo prior', 'coverage'))} / "
           f"{'n/m' if ss is None else f'{ss:.2f}'} |",
           f"| Answer at genus or family when unsure: accuracy at reported rank | - | - | - | "
           f"{cell(rv('vl600', '+ geo prior', 'backoff_acc'), rci('vl600', '+ geo prior', 'backoff_acc'))} |",
           f"| Open-set \"not in catalogue\" AUROC (61 unseen species) | - | - | - | "
           f"{cell(dig(fin, 'extras.vl600.open_set.auroc'))} |",
           f"| New species from 5 photos, no retraining: top-1 on 61 new species | names only, not measured | "
           f"needs retraining | not measured | {cell((fs.get(5) or {}).get('top1_new'))} |",
           f"| Multi-photo fusion (up to 3 photos): top-1 | - | - | - | "
           f"{cell(rv('vl600', 'VanaLens up to 3 photos + geo', 'top1'), rci('vl600', 'VanaLens up to 3 photos + geo', 'top1'))} |",
           "| Evidence shown with the answer | - | - | nearest photos | nearest photos + matched traits |",
           "| Uses location | - | - | - | GBIF prior |"]
    return "\n".join(out)


# Capability matrix of prior work. Only what each paper (or product page) states; "n/r" = not reported.
PRIOR = [
    # name, species, India, field photos, calibrated, sets, rejects unknown, add w/o retraining, evidence, location, multi-photo
    ("Leafsnap [R1]", "184 trees", "no (US)", "leaf, plain bg", "n/r", "ranked list", "non-leaf only", "n/r", "n/r",
     "n/r", "n/r"),
    ("Pl@ntNet-300K models [R2]", "1,081", "n/r", "yes", "n/r", "set metrics", "no", "retrain", "no", "no", "no"),
    ("Indian medicinal CNN [R3]", "6", "yes", "n/r", "n/r", "no", "no", "retrain", "no", "no", "no"),
    ("DIMPSAR, MED117, HerbLife [R4-R6]", "40-117", "yes", "DIMPSAR yes; MED117 no", "(data)", "(data)", "(data)",
     "(data)", "(data)", "(data)", "(data)"),
    ("Pl@ntNet app [R7]", "n/r", "n/r", "yes", "n/r", "ranked list", "n/r", "n/r", "n/r", "n/r", "n/r"),
    ("Google Lens, Seek", "n/r", "n/r", "yes", "n/r", "n/r", "n/r", "vendor", "n/r", "n/r", "n/r"),
    ("BioCLIP 2 zero-shot [R9]", "952K taxa (train)", "n/r", "yes", "n/r", "no", "no", "by name", "no", "no", "no"),
    ("PlantCLEF 2025 DINOv2 + head [R13]", "7,806", "no (Europe)", "quadrats", "n/r", "multi-label", "no", "no", "no",
     "no", "tiles"),
    ("VLMs in VLM4Bio [R14]", "no plants", "no", "yes", "n/r", "no", "no", "by name", "no", "no", "no"),
    ("Geo priors, SINR [R18, R19]", "part", "-", "-", "-", "-", "-", "-", "-", "yes", "-"),
    ("Conformal for plant ID [R16, R17]", "part", "-", "yes", "-", "yes", "no", "-", "-", "-", "yes"),
]
PRIOR_COLS = ["Species", "India", "Field photos", "Calibrated", "Sets", "Rejects unknown", "Add species, no retrain",
              "Evidence", "Location", "Multi-photo"]


def t_prior() -> str:
    fin = J("L6/final.json")
    fs5 = next((r for r in dig(fin, "extras.vl600.few_shot") or [] if r.get("shots") == 5), {})
    ece = dig(J("L3/ablation_vl600.json"), "rows.ensemble.ece")
    auc = val(dig(fin, "extras.vl600.open_set.auroc"))
    ours = ("**VanaLens (ours)**", "607", "yes", "yes", f"yes, ECE {pct(ece)}", "90% conformal",
            f"baseline gate, AUROC {pct(auc)}", f"{pct(val(fs5.get('top1_new')))} from 5 photos", "photos + traits",
            "GBIF prior", "up to 3")
    out = ["Table: Prior work against VanaLens [[compact]]. Cells state only what each paper or product page reports. "
           "n/r = not reported; (data) = dataset paper, no model; part = a component, not a full system; - = not "
           "applicable; bg = background. Codes in brackets point to the related-work references.", "", "| System | " + " | ".join(PRIOR_COLS) + " |", "|" + "---|" * (len(PRIOR_COLS) + 1)]
    for r in PRIOR + [ours]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


# Accuracies as reported by their authors, each on its own dataset. Never mixed with our numbers.
REPORTED = [
    ("Pl@ntNet-300K ViT-B (fine-tuned), repo README [R2]", "Pl@ntNet-300K test, 1,081 species", "top-1", "84.6%"),
    ("MobileNet, Kavitha et al. 2024 [R3]", "6 Indian medicinal species, 500 images each (Kaggle)", "accuracy", "98.3%"),
    ("Multi-organ CNN, Rzanny et al. 2019 [R8]", "101 species, 5 fixed views per plant",
     "top-1", "77% whole plant; 88% flower front; 97% all views fused"),
    ("BioCLIP 2, Gu et al. 2025 [R9]", "its zero-shot species benchmarks", "gain over BioCLIP", "+18.0 points"),
    ("GPT-4V in VLM4Bio, Maruf et al. 2024 [R14]", "Bird-10K species", "open-ended / multiple choice",
     "17.4% / 82.6% (BioCLIP open: 67.1%)"),
    ("DS@GT, frozen DINOv2 + tiling, 2nd in PlantCLEF 2025 [R13]", "PlantCLEF 2025 quadrats, multi-label",
     "macro F1 per quadrat (private)", "0.348"),
    ("DIMPSAR, MED117, HerbLife [R4-R6]", "dataset papers", "-", "no accuracy reported"),
]


def t_reported() -> str:
    out = ["Table: Accuracies reported by prior work, each on its own dataset and protocol. These numbers are NOT "
           "comparable with each other or with ours; they show the range of settings in the literature. Our numbers "
           "on our test split are in Table H2H.", "",
           "| Work | Dataset (theirs) | Metric | Reported |", "|---|---|---|---|"]
    for r in REPORTED:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


TABLES = {
    "prior": t_prior, "reported": t_reported, "h2h": t_h2h, "caps": t_caps,
    "main_vl600": lambda: t_main("vl600"), "main_inat": lambda: t_main("inat_india"),
    "ablation_vl600": lambda: t_ablation("vl600"), "ablation_inat": lambda: t_ablation("inat_india"),
    "uncertainty": t_uncertainty, "openset": t_openset, "fewshot": t_fewshot, "target": t_target,
    "dataset": t_dataset,
}


# ------------------------------------------------------------------ figures
def setup_mpl() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})


def fig_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    row1 = [("Photo(s)\n+ GPS (optional)", False), ("Plant check\nBioCLIP 2 zero-shot\norgan / not a plant", False),
            ("Frozen encoders\nBioCLIP 2 (768-d)\n+ DINO (1024-d)", True),
            ("Scorers\nprobe, MLP, kNN,\ntext traits", True)]
    row2 = [("Log-linear fusion\n+ temperature\n(fit on cal)", True), ("Location prior\nGBIF counts\naround the GPS", False),
            ("Multi-photo\nmean of log-probs\n(up to 3 photos)", False),
            ("Conformal 90% set\nopen-set check\nspecies/genus/family", True),
            ("Answer\n+ nearest reference\nphotos + traits", False)]
    w, h = 1.7, 1.15

    def draw(items, y, x0, gap):
        centers = []
        for i, (txt, key) in enumerate(items):
            x = x0 + i * (w + gap)
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                        fc="#e8f5e9" if key else "#f3f4f6", ec=GREEN if key else MUTED, lw=1.2))
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.6, color=INK)
            centers.append((x, y))
        for (xa, ya), (xb, _) in zip(centers, centers[1:]):
            ax.add_patch(FancyArrowPatch((xa + w + 0.02, ya + h / 2), (xb - 0.02, ya + h / 2),
                                         arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.1))
        return centers

    c1 = draw(row1, 2.25, 0.35, 0.75)
    c2 = draw(row2, 0.3, 0.05, 0.3)
    xa, ya = c1[-1]
    xb, yb = c2[0]
    ymid = (ya + yb + h) / 2
    ax.plot([xa + w / 2, xa + w / 2, xb + w / 2], [ya - 0.02, ymid, ymid], color=MUTED, lw=1.1)
    ax.add_patch(FancyArrowPatch((xb + w / 2, ymid), (xb + w / 2, yb + h + 0.02), arrowstyle="-|>",
                                 mutation_scale=11, color=MUTED, lw=1.1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def hbars(path: Path, labels, vals, cis, accent, xlabel, xmax=100.0, target=None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(labels) + 0.9))
    ys = list(range(len(labels)))[::-1]
    for y, v, c, a in zip(ys, vals, cis, accent):
        if v is None:
            ax.text(1, y, NM, va="center", color=MUTED, fontsize=9)
            continue
        ax.barh(y, 100 * v, color=GREEN if a else GREY, height=0.62)
        if c:
            ax.errorbar(100 * v, y, xerr=[[100 * (v - c[0])], [100 * (c[1] - v)]], fmt="none", ecolor=INK,
                        capsize=3, lw=1)
        ax.text(100 * (c[1] if c else v) + 0.8, y, f"{100 * v:.1f}%", va="center", fontsize=9, fontweight="bold")
    if target:
        ax.axvline(target, ls="--", color=INK, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_accuracy(path: Path) -> None:
    b2 = J("L2/b2_vl600.json") or {}
    items = [("BioCLIP 2 zero-shot, open-ended", b2.get("tol_top1"), None, False),
             ("BioCLIP 2 zero-shot, 607 names", rv("vl600", "B2 BioCLIP 2 kNN", "top1"), None, False),
             ("Linear probe (B3)", rv("vl600", "B3 linear probe", "top1"), rci("vl600", "B3 linear probe", "top1"), False),
             ("VanaLens, 1 photo", rv("vl600", "VanaLens 1 photo (no geo)", "top1"),
              rci("vl600", "VanaLens 1 photo (no geo)", "top1"), True),
             ("VanaLens, 1 photo + location", rv("vl600", "+ geo prior", "top1"), rci("vl600", "+ geo prior", "top1"), True),
             ("VanaLens, up to 3 photos + location", rv("vl600", "VanaLens up to 3 photos + geo", "top1"),
              rci("vl600", "VanaLens up to 3 photos + geo", "top1"), True)]
    hbars(path, [i[0] for i in items], [i[1] for i in items], [i[2] for i in items], [i[3] for i in items],
          "Top-1 accuracy on vl600 test (%)", xmax=108, target=95)


def fig_h2h(path: Path) -> None:
    """Head-to-head top-1 on vl600 test with 95% CI error bars (baselines grey, VanaLens green)."""
    rows = h2h_rows()
    hbars(path, [r["label"] for r in rows], [r["top1"][0] for r in rows], [r["top1"][1] for r in rows],
          [r["ours"] for r in rows], "Top-1 accuracy on the same vl600 test split (%)", xmax=108, target=95)


def fig_calibration(path: Path) -> None:
    rows = (J("L3/ablation_vl600.json") or {}).get("rows") or {}
    b2 = J("L2/b2_vl600.json") or {}
    items = [("Linear probe alone", dig(rows, "probe.ece"), False), ("MLP head alone", dig(rows, "mlp.ece"), False),
             ("Fused, before temperature", dig(rows, "fused.ece"), False),
             ("BioCLIP 2 zero-shot", b2.get("ece"), False), ("Fused + temperature", dig(rows, "fused+T.ece"), True),
             ("Final ensemble + temperature", dig(rows, "ensemble.ece"), True)]
    vals = [i[1] for i in items]
    top = max([v for v in vals if v is not None] or [0.1])
    hbars(path, [i[0] for i in items], vals, [None] * len(items), [i[2] for i in items],
          "Expected calibration error on vl600 test (%), lower is better", xmax=100 * top * 1.25)


def fig_fewshot(path: Path) -> None:
    fin = J("L6/final.json")
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for ds, col, lab in (("vl600", GREEN, "vl600 (61 new species)"), ("inat_india", "#5c6bc0", "inat_india (5 new species)")):
        fs = dig(fin, f"extras.{ds}.few_shot") or []
        if not fs:
            continue
        x = [r["shots"] for r in fs]
        v = [100 * val(r["top1_new"]) for r in fs]
        lo = [100 * r["top1_new"]["ci"][0] for r in fs]
        hi = [100 * r["top1_new"]["ci"][1] for r in fs]
        ax.plot(x, v, "o-", color=col, label=lab)
        ax.fill_between(x, lo, hi, color=col, alpha=0.15)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 2.5, f"{vi:.0f}", ha="center", fontsize=8.5, color=col)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 5, 10])
    ax.set_xticklabels(["1", "3", "5", "10"])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Gallery photos added per new species (no retraining)")
    ax.set_ylabel("Top-1 on new species (%)")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_photos(path: Path) -> str:
    """2x2 montage of the deck's vl600 photos; returns the credit line."""
    ims = []
    for k in PHOTO_SET:
        p = PITCH / "photos" / f"{k}_1.00.jpg"
        if p.exists():
            ims.append(Image.open(p).convert("RGB").resize((420, 420)))
    if not ims:
        return ""
    W = Image.new("RGB", (2 * 420 + 12, ((len(ims) + 1) // 2) * 432 - 12), "white")
    for i, im in enumerate(ims):
        W.paste(im, ((i % 2) * 432, (i // 2) * 432))
    W.save(path, quality=88)
    meta: dict[str, dict] = {}
    want = {v[0]: k for k, v in PHOTO_SET.items()}
    mp = DATA / "vl600" / "meta.csv"
    if mp.exists():
        with mp.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("image") in want:
                    meta[want[r["image"]]] = r
    parts = [f"<i>{PHOTO_SET[k][1]}</i>" + (f" (obs. {meta[k]['obs_id']}, {meta[k]['license'].upper()})" if k in meta else "")
             for k in PHOTO_SET]
    return "Photos from iNaturalist observers: " + "; ".join(parts) + "."


def figures() -> dict[str, tuple[str, str]]:
    """name -> (relative image path, caption). Generated figures are rebuilt from runs/ every time."""
    FIGS.mkdir(parents=True, exist_ok=True)
    setup_mpl()
    fig_pipeline(FIGS / "pipeline.png")
    fig_accuracy(FIGS / "accuracy.png")
    fig_h2h(FIGS / "h2h.png")
    fig_calibration(FIGS / "calibration.png")
    fig_fewshot(FIGS / "fewshot.png")
    credit = fig_photos(FIGS / "photos.jpg")
    figs = {
        "pipeline": ("figs/pipeline.png", "The VanaLens pipeline. Green boxes hold the parts fitted on the calibration "
                     "split (heads, fusion weights, temperature, conformal threshold); the two encoders stay frozen."),
        "accuracy": ("figs/accuracy.png", "Top-1 accuracy on the vl600 test split (post-cutoff observations). Error bars: "
                     "95% bootstrap CI over observations (baselines have point values only). Dashed line: the 95% target."),
        "h2h": ("figs/h2h.png", "Head-to-head on the same vl600 test split: BioCLIP 2 zero-shot, single-scorer and "
                "single-encoder baselines (grey) against VanaLens (green). Error bars: 95% bootstrap CI over "
                "observations; rows without bars have no per-photo predictions. "
                + ("Rows marked not measured are filled by the baselines run. "
                   if any(r["source"] == "pending L6-e" for r in h2h_rows("vl600")) else "")
                + "Dashed line: the 95% target."),
        "calibration": ("figs/calibration.png", "Calibration error before and after temperature scaling (vl600 test)."),
        "fewshot": ("figs/fewshot.png", "Accuracy on species added after training, by number of photos added. "
                    "Shaded: 95% CI. No weights are retrained."),
        "photos": ("figs/photos.jpg", "Example vl600 test-style photos: the system sees ordinary phone photos of "
                   "flowers, leaves and whole plants, not lab images. " + credit),
    }
    app = PITCH / "app" / "result.png"
    if app.exists():
        figs["app"] = ("../app/result.png", "What the user sees (screenshot from the demo app): the species, its "
                       "calibrated confidence, the 90% set, the visual traits that matched and the nearest reference "
                       "photos from the gallery.")
    return figs


# ------------------------------------------------------------------ placeholders
def fill(text: str, figs: dict[str, tuple[str, str]], log: dict[str, str]) -> str:
    d = derived()
    fig_no = {"n": 0}
    tab_names = list(TABLES)

    def rep(m: re.Match) -> str:
        key = m.group(1)
        kind, _, rest = key.partition(":")
        try:
            if kind == "row":
                ds, method, col = rest.split(":")
                s = pct(rv(ds, method, col))
            elif kind == "rowci":
                ds, method, col = rest.split(":")
                s = pct_ci(rv(ds, method, col), rci(ds, method, col))
            elif kind in ("pct", "pctci"):
                f, path = rest.split(":", 1)
                x = dig(J(f), path)
                s = pct(x) if kind == "pct" else pct_ci(x)
            elif kind == "num":
                f, path, fmt = rest.split(":", 2)
                x = val(dig(J(f), path))
                s = NM if x is None else format(x, fmt)
            elif kind == "d":
                s = d.get(rest, NM)
            elif kind == "table":
                if rest not in tab_names:
                    raise KeyError(rest)
                s = TABLES[rest]()
                s = fill(s, figs, log)   # tables may hold {{d:...}}
                return s
            elif kind == "fig":
                if rest not in figs:
                    return f"<p class='missing'>Figure {rest}: {NM}</p>"
                fig_no["n"] += 1
                src, cap = figs[rest]
                log[f"fig:{rest}"] = f"Figure {fig_no['n']}"
                return (f"\n<figure id='fig-{rest}'><img src='{src}' alt='{html.escape(rest)}'>"
                        f"<figcaption><b>Figure {fig_no['n']}.</b> {cap}</figcaption></figure>\n")
            else:
                raise KeyError(kind)
        except (ValueError, KeyError) as e:
            raise SystemExit(f"bad placeholder {{{{{key}}}}}: {e}")
        log[key] = s
        return s

    return re.sub(r"\{\{([^{}]+)\}\}", rep, text)


# ------------------------------------------------------------------ markdown -> html
def inline(s: str) -> str:
    """Escape, then apply **bold**, *italic*, `code`, [text](url). Raw <span>/<i>/<b>/<sup> tags pass through."""
    keep: list[str] = []

    def stash(m: re.Match) -> str:
        keep.append(m.group(0))
        return f"\x00{len(keep) - 1}\x00"

    s = re.sub(r"</?(span|i|b|sup|sub|br)( [^>]*)?/?>", stash, s)
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r"<a href='\2'>\1</a>", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)


def md_to_html(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    tab_no = 0
    para: list[str] = []

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        st = ln.strip()
        if not st:
            flush()
            i += 1
            continue
        if st.startswith("<figure") or st.startswith("<p class=") or st.startswith("<div") or st.startswith("</div"):
            flush()
            out.append(st)
            i += 1
            continue
        if st == "<!-- pagebreak -->":
            flush()
            out.append("<div class='pb'></div>")
            i += 1
            continue
        if st.startswith("<!--"):
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", st)
        if m:
            flush()
            lvl, txt = len(m.group(1)), m.group(2)
            cls = ""
            if txt.endswith("{-}"):
                txt, cls = txt[:-3].strip(), " class='nonum'"
            out.append(f"<h{lvl}{cls}>{inline(txt)}</h{lvl}>")
            i += 1
            continue
        if st.startswith("Table:") or st.startswith("|"):
            flush()
            cap = ""
            if st.startswith("Table:"):
                cap = st[6:].strip()
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            tab_no += 1
            compact = "[[compact]]" in cap
            cap = cap.replace(" [[compact]]", "").replace("[[compact]]", "")
            h = ["<table class='compact'>" if compact else "<table>"]
            if cap:
                h.append(f"<caption><b>Table {tab_no}.</b> {inline(cap)}</caption>")
            if rows:
                h.append("<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>")
                for r in rows[2:]:
                    h.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
                h.append("</tbody>")
            h.append("</table>")
            out.append("".join(h))
            continue
        if re.match(r"^(-|\d+\.)\s+", st):
            flush()
            ordered = bool(re.match(r"^\d+\.", st))
            tag = "ol" if ordered else "ul"
            items: list[str] = []
            while i < len(lines) and lines[i].strip():
                s2 = lines[i].strip()
                mm = re.match(r"^(-|\d+\.)\s+(.*)$", s2)
                if mm:
                    items.append(mm.group(2))
                elif items:
                    items[-1] += " " + s2
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(t)}</li>" for t in items) + f"</{tag}>")
            continue
        if st.startswith(">"):
            flush()
            q = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q.append(lines[i].strip()[1:].strip())
                i += 1
            out.append("<div class='note'>" + inline(" ".join(q)) + "</div>")
            continue
        para.append(st)
        i += 1
    flush()
    body = "\n".join(out)
    # Table references written as "Table OPENSET" etc. are resolved by caption order: replace NAME tokens.
    return body


CSS = """
@page { size: A4; margin: 16mm 16mm 17mm 16mm;
  @bottom-center { content: counter(page); font: 9pt Georgia, serif; color: #666; } }
@page :first { @bottom-center { content: none; } }
html { font-size: 10pt; }
body { font-family: Georgia, 'Times New Roman', serif; font-variant-numeric: lining-nums; line-height: 1.4; color: #1b1b1b; margin: 0; }
h1 { font-size: 23pt; margin: 0 0 4pt; color: #1e4d22; }
h2, h3, h4 { font-family: 'Segoe UI', Arial, sans-serif; color: #1e4d22; break-after: avoid; }
body { counter-reset: h2; }
h2 { font-size: 14pt; margin: 16pt 0 5pt; counter-increment: h2; counter-reset: h3; border-bottom: 1px solid #c8d6c9; padding-bottom: 2pt; }
h2::before { content: counter(h2) "  "; }
h3 { font-size: 11.5pt; margin: 11pt 0 3pt; counter-increment: h3; }
h3::before { content: counter(h2) "." counter(h3) "  "; }
h2.nonum::before, h3.nonum::before { content: none; }
h2.nonum { counter-increment: none; }
p { margin: 0 0 6pt; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 6pt 0; padding-left: 18pt; } li { margin-bottom: 2pt; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 9pt; background: #f2f4f2; padding: 0 2pt; }
a { color: #1e5e24; text-decoration: none; }
figure { margin: 10pt 0 12pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 72mm; }
#fig-pipeline img { max-height: 66mm; } #fig-photos img { max-height: 58mm; } #fig-app img { max-height: 52mm; }
figcaption { font-size: 9pt; color: #333; text-align: left; margin-top: 4pt; line-height: 1.35; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 8.7pt; break-inside: avoid;
  font-family: 'Segoe UI', Arial, sans-serif; }
caption { caption-side: top; text-align: left; font-family: Georgia, serif; font-size: 9pt; color: #333;
  margin-bottom: 4pt; line-height: 1.35; }
thead th { border-top: 1.4px solid #1e4d22; border-bottom: 1px solid #1e4d22; text-align: left; padding: 3pt 4pt;
  background: #f1f7f1; }
td { border-bottom: 0.5px solid #d9dfd9; padding: 2.5pt 4pt; vertical-align: top; }
tbody tr:last-child td { border-bottom: 1.4px solid #1e4d22; }
table.compact { font-size: 7.4pt; } table.compact td, table.compact th { padding: 1.8pt 3pt; }
.ci { color: #5f6b62; font-size: 7.8pt; white-space: nowrap; }
.note { border-left: 3px solid #2e7d32; background: #f3f8f3; padding: 5pt 8pt; margin: 6pt 0 9pt; font-size: 9.8pt; }
.pb { break-after: page; }
.titleblock { margin-top: 40mm; } .subtitle { font-size: 13pt; color: #444; margin-bottom: 18pt; }
.meta { font-size: 10pt; color: #555; margin-bottom: 26pt; }
.abstract { font-size: 10.2pt; border-top: 1px solid #c8d6c9; border-bottom: 1px solid #c8d6c9; padding: 8pt 0; }
.missing { color: #a33; }
"""


def resolve_table_refs(body: str) -> str:
    """'Table OPENSET' style references -> 'Table N', N = order of the generated table's caption."""
    order = []
    for m in re.finditer(r"<caption><b>Table (\d+)\.</b> ([^<]{0,60})", body):
        order.append((int(m.group(1)), m.group(2)))
    keys = {"DATASET": "two evaluation datasets", "MAINVL": "Test results on vl600", "MAININ": "Test results on inat_india",
            "ABLVL": "Ablation on vl600", "ABLIN": "Ablation on inat_india", "UNC": "Uncertainty outputs",
            "OPENSET": "Open-set test", "FEWSHOT": "Adding species", "TARGET": "Which headline",
            "PRIOR": "Prior work against", "REPORTED": "Accuracies reported", "H2H": "Head-to-head",
            "CAPS": "Capabilities beyond"}
    for k, start in keys.items():
        n = next((no for no, cap in order if cap.startswith(start) or start in cap), None)
        body = body.replace(f"Table {k}", f"Table {n}" if n else "a table below")
    return body


def resolve_fig_refs(body: str, log: dict[str, str]) -> str:
    def rep(m):
        return log.get(f"fig:{m.group(1)}", "a figure")
    return re.sub(r"Figure@(\w+)", rep, body)


def build_html() -> Path:
    figs = figures()
    log: dict[str, str] = {}
    md = re.sub(r"<!--(?! pagebreak).*?-->", "", SRC.read_text(encoding="utf-8"), flags=re.S)
    filled = fill(md, figs, log)
    if "\u2014" in filled:
        raise SystemExit("em dash found in the report; replace it")
    body = md_to_html(filled)
    body = resolve_table_refs(body)
    body = resolve_fig_refs(body, log)
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>VanaLens project report</title>"
           f"<style>{CSS}</style></head><body>{body}</body></html>")
    OUT_HTML.write_text(doc, encoding="utf-8")
    (HERE / "numbers.json").write_text(json.dumps({"built_from": "runs/", "post_dino_fix": post_fix(),
                                                   "values": log}, indent=1), encoding="utf-8")
    return OUT_HTML


def find_edge() -> str | None:
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if Path(p).exists():
            return p
    return shutil.which("msedge")


def build_pdf(html_path: Path) -> Path:
    edge = find_edge()
    if not edge:
        raise SystemExit("msedge not found: open VanaLens_report.html in a browser and print to PDF")
    with tempfile.TemporaryDirectory() as prof:
        cmd = [edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={prof}",
               "--run-all-compositor-stages-before-draw", "--virtual-time-budget=5000",
               f"--print-to-pdf={OUT_PDF}", html_path.resolve().as_uri()]
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    return OUT_PDF


def pdf_pages(path: Path) -> int:
    data = path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page[^s]", data))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)
    h = build_html()
    print(f"html: {h.relative_to(REPO)}  post_dino_fix={post_fix()}")
    if not a.no_pdf:
        p = build_pdf(h)
        print(f"pdf: {p.relative_to(REPO)}  pages={pdf_pages(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
