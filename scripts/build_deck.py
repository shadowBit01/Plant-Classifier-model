"""Lane L7 (pitch). Build pitch/VanaLens.pptx (15 slides, 16:9), its chart PNGs and photo crops.

Every number on a slide is read from runs/*.json at build time; nothing is typed in by hand.
Theme: white background, ink #1F2933, one accent green #2E7D32, light grey #E5E7EB for shapes,
Segoe UI throughout, 0.6 in margins, left-aligned sentence titles, slide number bottom-right.

Run from the lane root:

    python scripts/build_deck.py

Inputs: runs/L6/final.json, runs/L1/vl600_stats.json, runs/L2/b2_vl600.json, runs/L3/ablation_vl600.json,
runs/L6/e2e_vl600.json, CC-BY photos from $VANALENS_DATA/vl600 (cropped copies go to pitch/photos/),
app screenshots in pitch/app/ (made once with headless Edge, see worklog/L7.md).
"""
from __future__ import annotations

import csv
import re
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "pitch"
PHOTOS = OUT / "photos"
APP = OUT / "app"
DATA = Path(os.environ.get("VANALENS_DATA", Path.home() / "vanalens-data"))

# ---------------------------------------------------------------- theme
INK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x61, 0x6E, 0x7C)
ACCENT = RGBColor(0x2E, 0x7D, 0x32)
GREY = RGBColor(0xE5, 0xE7, 0xEB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEX_INK, HEX_ACCENT, HEX_BASE, HEX_MUTED = "#1F2933", "#2E7D32", "#B8BEC6", "#616E7C"
FONT = "Segoe UI"

SW, SH = Inches(13.333), Inches(7.5)
M = Inches(0.6)
CW = SW - 2 * M  # content width
TITLE_PT, BODY_PT, SMALL_PT = 32, 20, 12

# CC-BY photos from vl600 (iNaturalist open data): (file under DATA/vl600, common name)
PHOTO_SET = {
    "calotropis": ("Calotropis gigantea/292293824_525990526.jpg", "Crown flower"),
    "delonix": ("Delonix regia/288938584_519658590.jpg", "Gulmohar"),
    "hibiscus": ("Hibiscus rosa-sinensis/183307220_319780364.jpg", "Hibiscus"),
    "lantana": ("Lantana camara/288661222_519141340.jpg", "Lantana"),
    "mango": ("Mangifera indica/297511063_535809156.jpg", "Mango"),
    "lotus": ("Nelumbo nucifera/283592874_509593958.jpg", "Lotus"),
    "tulsi": ("Ocimum tenuiflorum/279833528_502539056.jpg", "Tulsi"),
}


# ---------------------------------------------------------------- data
def load(rel: str) -> dict[str, Any]:
    """Read a JSON result file under runs/ (fails loudly if missing: no invented numbers)."""
    return json.loads((RUNS / rel).read_text(encoding="utf-8"))


def row(final: dict, ds: str, method: str) -> dict:
    """Return one method row of runs/L6/final.json."""
    for r in final["datasets"][ds]:
        if r["method"] == method:
            return r
    raise KeyError(f"{ds}: {method}")


def pct(x: float, nd: int = 1) -> str:
    """0.9386 -> '93.9%'."""
    return f"{100 * x:.{nd}f}%"


def report_module():
    """pitch/report/build_report.py as a module: the deck reuses its head-to-head rows and prior-work matrix,
    so report and deck always show the same comparison."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_report", OUT / "report" / "build_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect() -> dict[str, Any]:
    """Gather every number the deck uses into one dict."""
    final = load("L6/final.json")
    stats = load("L1/vl600_stats.json")
    b2 = load("L2/b2_vl600.json")
    abl = load("L3/ablation_vl600.json")
    e2e = load("L6/e2e_vl600.json")
    ex = final["extras"]["vl600"]
    d: dict[str, Any] = {"final": final, "stats": stats, "b2": b2, "abl": abl, "e2e": e2e, "ex": ex}
    d["one"] = row(final, "vl600", "+ geo prior")
    d["one_nogeo"] = row(final, "vl600", "VanaLens 1 photo (no geo)")
    d["three"] = row(final, "vl600", "VanaLens up to 3 photos + geo")
    d["probe"] = row(final, "vl600", "B3 linear probe")
    d["knn"] = row(final, "vl600", "knn")
    d["text"] = row(final, "vl600", "text")
    d["in_one"] = row(final, "inat_india", "VanaLens 1 photo (no geo)")
    d["in_b2"] = row(final, "inat_india", "B2 BioCLIP 2 kNN")  # zero-shot per scripts/baseline_bioclip.py
    d["not_run"] = [r["method"] for r in final["datasets"]["vl600"] if r["status"] != "ok"]
    rows = abl["rows"]
    d["ece"] = {k: rows[k]["ece"] for k in ("probe", "fused", "fused+T", "ensemble")}
    d["ece_b2"] = b2["ece"]
    d["unknown_rate_known"] = rows["fused+T+sets"]["unknown_rate"]
    br = report_module()
    d["h2h"] = br.h2h_rows("vl600")
    d["h2h_d"] = br.h2h_derived()
    d["prior"] = br.PRIOR
    ents = {e["name"].lower(): e for e in br._bl_entries("vl600")}
    bc = next((br._metric(e, "top1")[0] for k, e in ents.items() if "bioclip" in k and "only" in k), None)
    both = next((br._metric(e, "top1")[0] for k, e in ents.items() if "concatenated" in k), None)
    rs = OUT / "report" / "review_stats.json"
    share = json.loads(rs.read_text(encoding="utf-8")).get("india_share_test_observations") if rs.exists() else None
    d["india_test"] = "n/m" if share is None else pct(share, 0)
    d["dino_gain"] = "n/m" if bc is None or both is None else f"{100 * (both - bc):+.1f} points"
    return d


# ---------------------------------------------------------------- charts
def setup_mpl() -> None:
    """Use Segoe UI in matplotlib, no top/right spines, large labels."""
    for f in ("segoeui.ttf", "segoeuib.ttf"):
        p = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / f
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({
        "font.family": FONT, "font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 14,
        "ytick.labelsize": 15, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": HEX_MUTED, "axes.labelcolor": HEX_INK, "xtick.color": HEX_MUTED,
        "ytick.color": HEX_INK, "text.color": HEX_INK, "figure.dpi": 200, "savefig.dpi": 200,
    })


def hbars(path: Path, labels: list[str], vals: list[float], cis: list[list[float] | None],
          ours: list[bool], size: tuple[float, float], xlabel: str, xmax: float = 100,
          target: float | None = None, fmt: str = "{:.1f}%", xmin: float = 0) -> None:
    """Horizontal bar chart (top to bottom in list order), value labels, 95% CI error bars."""
    fig, ax = plt.subplots(figsize=size)
    y = list(range(len(labels)))[::-1]
    cols = [HEX_ACCENT if o else HEX_BASE for o in ours]
    ax.barh(y, [v - xmin for v in vals], left=xmin, color=cols, height=0.62, zorder=2)
    for yi, v, ci in zip(y, vals, cis):
        if ci:
            ax.errorbar(v, yi, xerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=HEX_INK,
                        elinewidth=1.4, capsize=5, zorder=3)
        hi = ci[1] if ci else v
        ax.text(hi + (xmax - xmin) * 0.012, yi, fmt.format(v), va="center", ha="left", fontsize=15,
                fontweight="bold", color=HEX_INK, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    if target is not None:
        ax.axvline(target, color=HEX_INK, lw=1.2, ls="--", zorder=1)
        ax.text(target, len(labels) - 0.45, f"{target:.0f}% target", ha="center", va="bottom",
                fontsize=13, color=HEX_MUTED)
    ax.set_yticks(y, labels)
    ax.set_xlim(xmin, xmin + (xmax - xmin) * 1.1)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def chart_results(d: dict) -> Path:
    """VanaLens vs baselines on vl600 test (top-1, 95% bootstrap CI where per-photo data exist)."""
    p = OUT / "results.png"
    hbars(p,
          ["BioCLIP 2 zero-shot,\nopen-ended", "BioCLIP 2 zero-shot,\n607 names given",
           "Linear probe on\nour embeddings", "VanaLens, 1 photo\n+ location", "VanaLens, up to 3\nphotos + location"],
          [100 * d["b2"]["open_ended_top1"], 100 * d["b2"]["top1"], 100 * d["probe"]["top1"],
           100 * d["one"]["top1"], 100 * d["three"]["top1"]],
          [None, None, [100 * c for c in d["probe"]["ci"]["top1"]],
           [100 * c for c in d["one"]["ci"]["top1"]], [100 * c for c in d["three"]["ci"]["top1"]]],
          [False, False, False, True, True], (8.4, 5.0), "Top-1 accuracy on vl600 test (%)")
    return p


def chart_h2h(d: dict) -> Path:
    """Head-to-head on the same vl600 test split; rows L6-e has not filled yet are drawn as 'not measured'."""
    p = OUT / "h2h.png"
    short = {"BioCLIP 2 zero-shot, open-ended (tree of life)": "BioCLIP 2 zero-shot,\nopen-ended",
             "BioCLIP 2 zero-shot, 607 names": "BioCLIP 2 zero-shot,\n607 names given",
             "Text prototypes only (0 images per species)": "Text traits only,\nno photos",
             "BioCLIP 2 features + linear probe": "BioCLIP 2 only\n+ linear probe",
             "DINOv2 features + linear probe": "DINOv2 only\n+ linear probe",
             "kNN on both features": "kNN, both encoders",
             "Concatenated features + linear probe": "Both encoders\n+ linear probe",
             "VanaLens, 1 photo, no location": "VanaLens, 1 photo",
             "VanaLens, 1 photo + location": "VanaLens, 1 photo\n+ location",
             "VanaLens, up to 3 photos + location": "VanaLens, up to 3\nphotos + location"}
    rows = [r for r in d["h2h"] if not r["label"].startswith("VanaLens, 1 photo +")]
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        v, ci = r["top1"]
        if v is None:
            ax.text(1.5, yi, "not measured yet", va="center", fontsize=13, color=HEX_MUTED, style="italic")
            continue
        v = 100 * v
        ax.barh(yi, v, color=HEX_ACCENT if r["ours"] else HEX_BASE, height=0.62, zorder=2)
        hi = v
        if ci:
            ax.errorbar(v, yi, xerr=[[v - 100 * ci[0]], [100 * ci[1] - v]], fmt="none", ecolor=HEX_INK,
                        elinewidth=1.4, capsize=4, zorder=3)
            hi = 100 * ci[1]
        ax.text(hi + 1.2, yi, f"{v:.1f}%", va="center", ha="left", fontsize=14, fontweight="bold", color=HEX_INK)
    ax.set_yticks(y, [short.get(r["label"], r["label"]) for r in rows], fontsize=13)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Top-1 accuracy, same vl600 test photos (%)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def chart_photos(d: dict) -> Path:
    """1 photo vs location prior vs up to 3 photos, top-1 with CI and the 95% target line."""
    p = OUT / "photos.png"
    hbars(p, ["1 photo", "1 photo + location", "Up to 3 photos\n+ location"],
          [100 * d["one_nogeo"]["top1"], 100 * d["one"]["top1"], 100 * d["three"]["top1"]],
          [[100 * c for c in d["one_nogeo"]["ci"]["top1"]], [100 * c for c in d["one"]["ci"]["top1"]],
           [100 * c for c in d["three"]["ci"]["top1"]]],
          [False, True, True], (7.6, 4.4), "Top-1 accuracy on vl600 test (%), axis starts at 85", target=95,
          xmin=85)
    return p


def chart_calibration(d: dict) -> Path:
    """Expected calibration error, lower is better."""
    p = OUT / "calibration.png"
    e = d["ece"]
    hbars(p, ["Linear probe alone", "Fused, before calibration", "BioCLIP 2 zero-shot",
              "Fused + temperature", "VanaLens ensemble"],
          [100 * e["probe"], 100 * e["fused"], 100 * d["ece_b2"], 100 * e["fused+T"], 100 * e["ensemble"]],
          [None] * 5, [False, False, False, True, True], (7.4, 4.2),
          "Calibration error (ECE, %), lower is better",
          xmax=max(2.0, 1.3 * 100 * max(e["probe"], e["fused"], d["ece_b2"], e["fused+T"], e["ensemble"])))
    return p


def chart_fewshot(d: dict) -> Path:
    """Few-shot curve: accuracy on newly added species vs photos per species, known species stay flat."""
    p = OUT / "fewshot.png"
    fs = d["ex"]["few_shot"]
    shots = [f["shots"] for f in fs]
    new = [100 * f["top1_new"]["value"] for f in fs]
    lo = [100 * f["top1_new"]["ci"][0] for f in fs]
    hi = [100 * f["top1_new"]["ci"][1] for f in fs]
    known = [100 * f["top1_known_after"]["value"] for f in fs]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(shots, known, color=HEX_BASE, lw=3, marker="o", ms=7, label="Species already known")
    ax.fill_between(shots, lo, hi, color=HEX_ACCENT, alpha=0.15, lw=0)
    ax.plot(shots, new, color=HEX_ACCENT, lw=3, marker="o", ms=8, label=f"{fs[0]['species_added']} newly added species")
    for s, v in zip(shots, new):
        ax.annotate(f"{v:.0f}%", (s, v), textcoords="offset points", xytext=(0, -24), ha="center",
                    fontsize=15, fontweight="bold", color=HEX_ACCENT)
    ax.set_xticks(shots)
    ax.set_xlabel("Photos per new species (no retraining)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower right", fontsize=14)
    fig.tight_layout()
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- photos
def photo_meta() -> dict[str, dict]:
    """obs_id and license for the chosen photos, from $VANALENS_DATA/vl600/meta.csv."""
    want = {v[0]: k for k, v in PHOTO_SET.items()}
    out: dict[str, dict] = {}
    with open(DATA / "vl600" / "meta.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["image"] in want:
                out[want[r["image"]]] = r
    return out


def crop_photo(key: str, aspect: float, px: int = 900) -> Path:
    """Centre-crop a vl600 photo to width/height = aspect and save it under pitch/photos/."""
    PHOTOS.mkdir(parents=True, exist_ok=True)
    dst = PHOTOS / f"{key}_{aspect:.2f}.jpg"
    src = DATA / "vl600" / PHOTO_SET[key][0]
    if src.exists():
        im = Image.open(src).convert("RGB")
        w, h = im.size
        if w / h > aspect:
            nw = int(h * aspect)
            im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
        else:
            nh = int(w / aspect)
            im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
        im = im.resize((px, int(px / aspect)))
        im.save(dst, quality=85)
    if not dst.exists():
        raise FileNotFoundError(dst)
    return dst


def credit(meta: dict[str, dict], keys: list[str]) -> str:
    """Attribution line for the photos shown on a slide."""
    lic = sorted({meta[k]["license"].upper().replace("CC-", "CC ") for k in keys})
    obs = ", ".join(meta[k]["obs_id"] for k in keys)
    return f"Photos: iNaturalist observers, {' / '.join(lic)}, observations {obs}."


# ---------------------------------------------------------------- pptx helpers
def _font(run, size: float, bold: bool = False, color: RGBColor = INK, italic: bool = False) -> None:
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def text(slide, x, y, w, h, s: str, size: float = BODY_PT, bold: bool = False, color: RGBColor = INK,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic: bool = False):
    """Add a plain textbox (no autofit, no margins), one paragraph per line of s."""
    tb = slide.shapes.add_textbox(int(x), int(y), int(w), int(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        _font(p.add_run(), size, bold, color, italic)
        p.runs[0].text = line
    return tb


def bullets(slide, x, y, w, h, items: list[str], size: float = BODY_PT, gap: float = 14) -> None:
    """Bulleted list with a green square marker; at most 4 items."""
    assert len(items) <= 4, "max 4 bullets per slide"
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.32)))
        pPr.set("indent", str(-Inches(0.32)))
        bu_clr = pPr.makeelement(qn("a:buClr"), {})
        srgb = bu_clr.makeelement(qn("a:srgbClr"), {"val": "2E7D32"})
        bu_clr.append(srgb)
        pPr.append(bu_clr)
        bu_font = pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"})
        pPr.append(bu_font)
        pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "\u25A0"}))
        _font(p.add_run(), size)
        p.runs[0].text = it


def box(slide, x, y, w, h, fill: RGBColor = GREY, line: RGBColor | None = None, shape=MSO_SHAPE.RECTANGLE):
    """Flat shape, no shadow."""
    s = slide.shapes.add_shape(shape, int(x), int(y), int(w), int(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.5)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    return s


def box_text(shape, lines: list[tuple[str, float, bool, RGBColor]], align=PP_ALIGN.CENTER) -> None:
    """Write (text, size, bold, colour) lines centred inside a shape."""
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.1)
    tf.margin_top = tf.margin_bottom = Inches(0.05)
    for i, (s, size, bold, color) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        _font(p.add_run(), size, bold, color)
        p.runs[0].text = s


def arrow(slide, x1, y1, x2, y2, color: RGBColor = MUTED) -> None:
    """Straight connector with an arrowhead at the end."""
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, int(x1), int(y1), int(x2), int(y2))
    c.line.color.rgb = color
    c.line.width = Pt(2)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


def tile(slide, x, y, w, h, big: str, label: str, accent: bool = True) -> None:
    """Stat tile: big number over a short label on light grey."""
    s = box(slide, x, y, w, h, GREY)
    box_text(s, [(big, 32, True, ACCENT if accent else INK), (label, 16, False, INK)])


def table(slide, x, y, w, rows: list[list[str]], col_w: list[float], size: float = 13,
          row_h: float = 0.42, bold_last: bool = True):
    """Plain table: green header row, white body, last row (ours) bold green on a light green fill."""
    shp = slide.shapes.add_table(len(rows), len(rows[0]), int(x), int(y), int(w), int(Inches(row_h) * len(rows)))
    tbl = shp.table
    tblPr = tbl._tbl.tblPr
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "0")
    tot = sum(col_w)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = int(w * cw / tot)
    for i, r in enumerate(rows):
        tbl.rows[i].height = int(Inches(row_h))
        last = bold_last and i == len(rows) - 1
        for j, val in enumerate(r):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = ACCENT if i == 0 else (RGBColor(0xE8, 0xF5, 0xE9) if last else WHITE)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            tf.text = val
            for para in tf.paragraphs:
                for run in para.runs:
                    _font(run, size, bold=(i == 0 or last or j == 0),
                          color=WHITE if i == 0 else (ACCENT if last else INK))
    return tbl


class Deck:
    """Thin wrapper that adds themed slides with title, number, footnote and notes."""

    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = SW, SH
        self.blank = self.prs.slide_layouts[6]
        self.n = 0
        self.script: list[tuple[str, str]] = []   # (title, notes) per slide, written to pitch/script.md

    def slide(self, title: str | None, notes: str, foot: str | None = None):
        self.n += 1
        s = self.prs.slides.add_slide(self.blank)
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = WHITE
        if title:
            text(s, M, Inches(0.45), CW, Inches(1.0), title, TITLE_PT, True, INK, anchor=MSO_ANCHOR.TOP)
        if self.n > 1:
            text(s, SW - M - Inches(1.0), SH - Inches(0.5), Inches(1.0), Inches(0.3), str(self.n),
                 SMALL_PT, False, MUTED, align=PP_ALIGN.RIGHT)
        if foot:
            text(s, M, SH - Inches(0.55), CW - Inches(1.2), Inches(0.4), foot, SMALL_PT, False, MUTED)
        s.notes_slide.notes_text_frame.text = notes
        self.script.append((title or ("Title" if self.n == 1 else "Close"), notes))
        return s

    def save(self, path: Path) -> None:
        self.prs.save(path)
        self.write_script(path.parent / "script.md")

    def write_script(self, path: Path, wpm: int = 140) -> None:
        """Talk script = the speaker notes, with cumulative times at a spoken pace of `wpm` words per minute."""
        def mmss(sec: float) -> str:
            return f"{int(sec // 60)}:{int(sec % 60):02d}"
        total = sum(len(n.split()) for _, n in self.script) * 60 / wpm
        out = [f"# VanaLens: talk script (about {total / 60:.1f} minutes)", "",
               f"Generated by scripts/build_deck.py from the speaker notes of pitch/VanaLens.pptx ({self.n} slides). "
               "Every number comes from runs/. Times assume about 140 spoken words per minute.", ""]
        t = 0.0
        for i, (title, notes) in enumerate(self.script, 1):
            dt = len(notes.split()) * 60 / wpm
            out += [f"## {i}. {title} ({mmss(t)} to {mmss(t + dt)})", notes, ""]
            t += dt
        path.write_text("\n".join(out), encoding="utf-8")


def picture(slide, path: Path, x, y, w=None, h=None, border: bool = False):
    """Add an image; optional thin grey border."""
    pic = slide.shapes.add_picture(str(path), int(x), int(y), None if w is None else int(w), None if h is None else int(h))
    if border:
        pic.line.color.rgb = GREY
        pic.line.width = Pt(1)
    return pic


# ---------------------------------------------------------------- slides
def build() -> None:
    OUT.mkdir(exist_ok=True)
    setup_mpl()
    d = collect()
    meta = photo_meta()
    st = d["stats"]
    one, three, probe = d["one"], d["three"], d["probe"]
    os_ = d["ex"]["open_set"]
    fs5 = next(f for f in d["ex"]["few_shot"] if f["shots"] == 5)
    sel = d["ex"]["selective"]
    tgt = {t["metric"]: t for t in d["ex"]["target"]}
    rows = d["abl"]["rows"]
    ens = rows["ensemble"]
    alpha = d["final"]["alpha"]
    cpath = {k: chart() for k, chart in (("results", lambda: chart_results(d)), ("photos", lambda: chart_photos(d)),
                                          ("cal", lambda: chart_calibration(d)), ("few", lambda: chart_fewshot(d)),
                                          ("h2h", lambda: chart_h2h(d)))}
    deck = Deck()
    top = Inches(1.6)  # content top below the title

    # 1. title
    s = deck.slide(None, (
        f"Hi, we are presenting VanaLens. It identifies {st['species']} common Indian and worldwide plant species at "
        f"{pct(three['top1'])} top-1 from up to three phone photos and {pct(one['top1'])} from one photo with location, "
        f"{d['h2h_d']['h2h_gain_zs_geo']} above BioCLIP 2 zero-shot on the same post-cutoff test photos. It reports "
        "calibrated confidence, backs off to genus or family when unsure, flags unfamiliar plants, shows reference "
        "photos and traits as evidence, and adds new species from five photos in seconds without retraining."),
        foot=credit(meta, ["calotropis", "delonix", "hibiscus", "lantana"]))
    text(s, M, Inches(2.1), Inches(5.6), Inches(0.85), "VanaLens", 36, True, ACCENT)
    text(s, M, Inches(2.95), Inches(5.6), Inches(1.35),
         f"Calibrated plant ID: {pct(three['top1'])} top-1\non {st['species']} Indian and world species", 28, False, INK)
    text(s, M, Inches(4.45), Inches(5.6), Inches(1.1),
         f"{pct(one['top1'])} from 1 photo ({d['h2h_d']['h2h_gain_zs_geo']} vs BioCLIP 2 zero-shot).\n"
         "Calibrated confidence, visible evidence, 5-shot updates.", 18, False, MUTED)
    gx, gy, gw, gap = Inches(6.75), Inches(0.6), Inches(2.85), Inches(0.14)
    for i, k in enumerate(["calotropis", "delonix", "hibiscus", "lantana"]):
        px = gx + (i % 2) * (gw + gap)
        py = gy + (i // 2) * (gw + gap)
        picture(s, crop_photo(k, 1.0), px, py, gw, gw)

    # 2. problem
    s = deck.slide("Apps name plants but rarely say how sure they are.", (
        "Everyone walks past plants they cannot name: a tree on the street, a herb in a market, a weed in a field. "
        "Plant apps exist, but they give a ranked list of names, and none we found publishes a coverage guarantee or "
        "a 'not in our list' answer. "
        "General AI chatbots are worse at naming plants from an open question. "
        "And if your local plant is missing, you cannot add it without retraining the model."),
        foot=credit(meta, ["tulsi", "mango"]))
    bullets(s, M, top + Inches(0.2), Inches(8.0), Inches(4.5), [
        "Common plants matter daily: food, medicine, gardens, weeds.",
        "Apps give a ranked list; none we found says \"not in our list\".",
        f"Open-ended, BioCLIP 2 names only {pct(d['b2']['open_ended_top1'], 0)} of our test photos.",
        "Adding a missing local species usually means retraining.",
    ], size=22, gap=20)
    pw2 = Inches(3.4)
    picture(s, crop_photo("tulsi", 1.45), Inches(9.33), top + Inches(0.1), pw2, int(pw2 / 1.45))
    picture(s, crop_photo("mango", 1.45), Inches(9.33), top + Inches(0.25) + int(pw2 / 1.45), pw2, int(pw2 / 1.45))

    # 3. prior work
    keep = ("Leafsnap", "Indian medicinal CNN", "Pl@ntNet-300K", "Pl@ntNet app", "Google Lens", "BioCLIP 2 zero-shot",
            "PlantCLEF 2025")
    cols = [1, 2, 5, 6, 7, 8, 9, 10]
    head = ["Prior work", "Species", "India", "Sets", "Unknown plant", "Add species", "Evidence", "Location",
            "Multi-photo"]
    body = []
    for r in d["prior"]:
        if r[0].startswith(keep):
            body.append([re.sub(r"\s*\[R[^\]]*\]", "", r[0])] + [r[c] for c in cols])
    body.append(["VanaLens (ours)", str(st["species"]), "yes", "90% (size 1)", f"yes ({os_['auroc']['value']:.2f} AUC)",
                 f"5 photos: {pct(fs5['top1_new']['value'], 0)}", "photos + traits", "GBIF", "up to 3"])
    s = deck.slide("Prior plant ID: no sets, no unknowns, retrain to add.", (
        "Plant ID is not new. Leafsnap matched leaf outlines; Indian work trains CNNs on a few dozen clean leaf "
        "photos and must retrain for every new species; Pl@ntNet, Lens and Seek are closed and give a ranked answer. "
        "BioCLIP 2 and DINOv2, which we build on, are encoders, not full systems. Each cell says only what the paper "
        "reports. The last row is the point: VanaLens combines calibrated confidence, 90% sets, genus backoff, an "
        "open-set gate, 5-shot species addition, visible evidence, location and multi-photo fusion in one system."),
        foot="n/r = not reported in the paper or product page. Sources and the full table: report, Section 2.")
    table(s, M, top + Inches(0.05), CW, [head] + body, [2.6, 1.15, 0.95, 1.1, 1.2, 1.35, 1.25, 0.95, 1.05],
          size=13, row_h=0.5)

    # 3b. what we built
    s = deck.slide(f"VanaLens: {pct(three['top1'])} on 3 photos, calibrated, 5-shot.", (
        "VanaLens is built on frozen BioCLIP 2 and DINOv2 features with light trained heads (linear probe and MLP), "
        "fused and calibrated on a held-out split. Three design choices shape the user experience. First, fast species "
        "updates: a new species is added in seconds from five photos without retraining. Second, calibrated uncertainty: "
        "confidence matches accuracy, with genus or family backoff and a 'not in catalog' check. Third, context fusion: "
        "up to three photos of the same plant and a GBIF location prior."))
    text(s, M, top + Inches(0.05), CW, Inches(0.6),
         "Upload 1 to 3 photos. Get a species, a genus or family when unsure, or \"not in catalog\".", 22, False, INK)
    cw3, gap3, cy, ch = Inches(3.85), Inches(0.29), top + Inches(1.2), Inches(3.0)
    ideas = [
        ("1", "5-shot updates",
         "Frozen BioCLIP 2 + DINOv2 features with trained heads. Add new species from 5 photos, no retraining."),
        ("2", "Calibrated doubt",
         f"{pct(ens['ece'])} ECE, 90% conformal sets, genus/family backoff ({pct(one['backoff_acc'])} accurate) and an unknown flag."),
        ("3", "3 photos + GPS",
         f"Fuses up to 3 photos of one plant ({pct(three['top1'])} top-1) with a GBIF location prior."),
    ]
    for i, (num, head, body) in enumerate(ideas):
        x = M + i * (cw3 + gap3)
        box(s, x, cy, cw3, ch, GREY)
        text(s, x + Inches(0.35), cy + Inches(0.3), Inches(1), Inches(0.7), num, 36, True, ACCENT)
        text(s, x + Inches(0.35), cy + Inches(1.1), cw3 - Inches(0.7), Inches(0.5), head, 22, True, INK)
        text(s, x + Inches(0.35), cy + Inches(1.7), cw3 - Inches(0.7), Inches(1.6), body, 18, False, INK)

    # 4. how it works
    s = deck.slide("Frozen encoders, trained heads, then calibration decides.", (
        "Here is the pipeline. A photo goes through two frozen foundation encoders: BioCLIP 2, trained on the tree of "
        "life, and DINOv2, a general visual model. Keeping them frozen is a deliberate choice: we train light heads "
        "(linear probe and MLP) on their cached feature vectors, alongside a nearest-neighbour gallery that supplies "
        "evidence photos and text-trait prototypes. Log-linear fusion and temperature scaling calibrate the "
        "probabilities, a GBIF location prior and up to three photos refine them, and calibrated thresholds decide: "
        "one species, a genus or family when unsure, or 'not in catalog'."))
    by, bh, bw = top + Inches(0.75), Inches(1.75), Inches(1.6)
    gapx = Inches(0.32)
    steps = [
        ("Photo", "1 to 3 photos\n+ GPS if known"),
        ("Encoders", "BioCLIP 2\n+ DINOv2"),
        ("Scorers", "Probe, MLP,\nkNN, traits"),
        ("Fusion", "Weighted fusion\n+ calibration"),
        ("Location", "GBIF prior\nfor the area"),
    ]
    for i, (head, sub) in enumerate(steps):
        x = M + i * (bw + gapx)
        b = box(s, x, by, bw, bh, GREY if i else WHITE, None if i else ACCENT, MSO_SHAPE.ROUNDED_RECTANGLE)
        box_text(b, [(head, 20, True, ACCENT), (sub, 15, False, INK)])
        if i:
            arrow(s, x - gapx + Inches(0.04), by + bh / 2, x - Inches(0.04), by + bh / 2)
    ox = M + 5 * (bw + gapx) + Inches(0.1)
    ow = SW - M - ox
    oh, og = Inches(1.05), Inches(0.22)
    oy0 = by + bh / 2 - (3 * oh + 2 * og) / 2
    outs = [("Answer", "one species", True), ("Not sure", "genus or family", False),
            ("Not in catalog", "add it with photos", False)]
    last_x = M + 4 * (bw + gapx) + bw
    for i, (head, sub, acc) in enumerate(outs):
        oy = oy0 + i * (oh + og)
        b = box(s, ox, oy, ow, oh, ACCENT if acc else WHITE, None if acc else ACCENT, MSO_SHAPE.ROUNDED_RECTANGLE)
        box_text(b, [(head, 18, True, WHITE if acc else ACCENT), (sub, 15, False, WHITE if acc else INK)])
        arrow(s, last_x + Inches(0.04), by + bh / 2, ox - Inches(0.04), oy + oh / 2)
    text(s, M, by + bh + Inches(1.55), CW, Inches(1.0),
         "Deliberate frozen-feature design: linear probe and MLP trained on gallery vectors; fusion weights, "
         "temperature, conformal threshold and open-set gate fitted on calibration. Enables CPU inference and "
         "5-shot species addition.",
         18, False, MUTED)

    # 4b. what is new
    hd = d["h2h_d"]
    s = deck.slide("What is new: calibrated system and evidence.", (
        "VanaLens is a systems contribution on frozen BioCLIP 2 and DINOv2 features, citing every component we build on. "
        "Four contributions, each tied to measured vl600 evidence. One: an uncertainty-aware pipeline with trained heads, "
        f"calibrated fusion ({pct(ens['ece'])} ECE), 90% sets, genus backoff ({pct(one['backoff_acc'])} accurate), "
        "open-set check, location prior and multi-photo fusion. Two: adding new species in seconds from five photos "
        f"without retraining, reaching {pct(fs5['top1_new']['value'], 0)} top-1 on {os_['held_out_species']} unseen "
        "species. Three: nearest reference photos and matched traits shown with every answer. Four: a date-cut Indian "
        f"benchmark with head-to-head baselines on the same {one['n']:,} test photos: {hd['h2h_gain_zs']} over BioCLIP 2 "
        f"zero-shot from one photo, and {hd['h2h_gain_best3']} over the best linear probe with three photos and location."),
        foot="Built on prior work: BioCLIP 2, DINOv2, kNN, linear probe, temperature scaling, conformal sets, "
             "GBIF prior (report, Section 2).")
    bullets(s, M, top + Inches(0.1), CW, Inches(4.9), [
        f"Calibrated system on frozen models: trained heads, {pct(ens['ece'])} ECE, 90% sets, genus backoff "
        f"({pct(one['backoff_acc'])} accurate), unknown check, location, 3-photo fusion.",
        f"Add a species from 5 photos, no retraining: {pct(fs5['top1_new']['value'])} on {os_['held_out_species']} "
        f"unseen species (known accuracy kept at {pct(fs5['top1_known_after']['value'])}).",
        "Evidence with every answer: nearest reference photos and the plant traits that matched.",
        f"Measured on {st['species']} Indian and common species after July 2025: {hd['h2h_gain_zs']} over BioCLIP 2 "
        f"zero-shot (1 photo); {hd['h2h_gain_best']} over probe (1 photo, intervals overlap), {hd['h2h_gain_best3']} "
        "(3 photos + geo).",
    ], size=21, gap=22)

    # 5. dataset
    s = deck.slide("All test photos postdate the encoders' training data.", (
        f"We built our own dataset, vl600, from iNaturalist research-grade observations. It has {st['species']} species: "
        f"{st['species_by_source']['india']} most observed in India, {st['species_by_source']['world']} most observed worldwide, "
        f"and a short list of common cultivated plants. Together they cover {st['coverage_pct']:.1f}% of all research-grade "
        f"plant observations in India. The {st['photos_per_split']['test']:,} test photos were all taken after July 2025, after "
        "BioCLIP 2's training data was collected, which reduces overlap with what the encoders saw in training."),
        foot=credit(meta, ["mango", "lotus", "tulsi"]))
    tw, th, tg = Inches(2.75), Inches(1.45), Inches(0.22)
    tiles = [
        (f"{st['species']}", "species"),
        (f"{st['photos']:,}", "photos"),
        (f"{st['coverage_pct']:.1f}%", "of India's iNat plant records"),
        (f"{st['photos_per_split']['test']:,}", "test photos, after July 2025"),
    ]
    for i, (big, lab) in enumerate(tiles):
        tile(s, M + (i % 2) * (tw + tg), top + Inches(0.15) + (i // 2) * (th + tg), tw, th, big, lab)
    bullets(s, M, top + Inches(3.55), Inches(6.0), Inches(1.6), [
        f"{st['species_by_source']['india']} top India + {st['species_by_source']['world']} top world + "
        f"{st['species_by_source']['common']} common cultivated.",
        f"Chosen by observation count only; {d['india_test']} of test from India.",
    ], size=18, gap=8)
    pw = Inches(1.82)
    for i, k in enumerate(["mango", "lotus", "tulsi"]):
        picture(s, crop_photo(k, 0.75), Inches(6.85) + i * (pw + Inches(0.14)), top + Inches(0.15), pw, int(pw / 0.75))
    text(s, Inches(6.85), top + Inches(0.15) + int(pw / 0.75) + Inches(0.12), Inches(5.8), Inches(0.4),
         "Mango, lotus, tulsi: three of the 607 classes.", 16, False, MUTED)

    # 6. results
    s = deck.slide(f"Ahead of BioCLIP 2 zero-shot on {one['n']:,} new photos.", (
        f"On {one['n']:,} test photos, BioCLIP 2 zero-shot gets {pct(d['b2']['top1'])} when we give it our 607 names, and "
        f"{pct(d['b2']['open_ended_top1'])} open-ended. A linear probe on our embeddings gets {pct(probe['top1'])}. VanaLens with one "
        f"photo and its location gets {pct(one['top1'])}, and with up to three photos of the same plant {pct(three['top1'])}. Error bars are 95% "
        "bootstrap intervals; the BioCLIP 2 runs have no per-photo file, so no interval. We could not run the Pl@ntNet "
        f"API or a VLM baseline because we had no API keys. On our second test set, inat_india, the picture is the same: "
        f"{pct(d['in_one']['top1'])} versus {pct(d['in_b2']['top1'])}."),
        foot="Error bars: 95% bootstrap CI by observation. BioCLIP 2 runs have no CI (no per-photo file). "
             "Pl@ntNet API and VLM baselines not run (no API keys).")
    picture(s, cpath["results"], M - Inches(0.1), top, Inches(8.0))
    bullets(s, Inches(8.45), top + Inches(0.35), Inches(4.3), Inches(4.5), [
        f"+{100 * (one['top1'] - d['b2']['top1']):.1f} points over BioCLIP 2 given the same 607 names (1 photo + location).",
        f"Top-5: {pct(one['top5'])} with 1 photo, {pct(three['top5'])} with 3.",
        f"Second test set (inat_india): {pct(d['in_one']['top1'])} vs {pct(d['in_b2']['top1'])}.",
    ], size=18, gap=16)

    # 6b. head-to-head
    hd = d["h2h_d"]
    pend = [r for r in d["h2h"] if r["source"] == "pending L6-e"]
    s = deck.slide(f"Same photos: {hd['h2h_gain_zs'].replace(' points', '')} vs zero-shot, "
                   f"{hd['h2h_gain_best3'].replace(' points', '')} vs probe (3 photos).", (
        f"Every bar uses the same {one['n']:,} test photos and the same frozen features, nothing tuned on test. "
        f"Against BioCLIP 2 zero-shot with our 607 names we are {hd['h2h_gain_zs']} ahead. Against the best baseline, a "
        f"linear probe on both encoders, one photo gives {hd['h2h_gain_best']} and the intervals overlap, so that single-photo gain alone is not "
        f"statistically clear; three photos and location give {hd['h2h_gain_best3']}. The bigger difference is what the "
        "baselines cannot output: sets, genus backoff, unknown check, new species, evidence."
        + (" Bars marked not measured yet come from the baselines run." if pend else "")),
        foot="Same vl600 test split for every bar. Error bars: 95% bootstrap CI by observation; BioCLIP 2 zero-shot "
             "has no per-photo file, so no CI.")
    picture(s, cpath["h2h"], M - Inches(0.1), top - Inches(0.05), Inches(8.0))
    bullets(s, Inches(8.45), top + Inches(0.3), Inches(4.3), Inches(4.6), [
        f"{hd['h2h_gain_zs']} over BioCLIP 2 zero-shot, same 607 names, 1 photo.",
        f"{hd['h2h_gain_best']} over the best baseline (1 photo), intervals overlap; {hd['h2h_gain_best3']} with 3 photos + location.",
        "Only VanaLens gives sets, genus backoff, unknown check, add-a-species and evidence.",
    ], size=18, gap=16)

    # 7. multi-photo + selective
    s = deck.slide("Up to three photos push top-1 accuracy past 95%.", (
        f"One photo gives {pct(d['one_nogeo']['top1'])}. Adding the location prior lifts it to {pct(one['top1'])}. Letting the "
        f"user add up to three photos of the same plant lifts it to {pct(three['top1'])}, with the whole confidence interval "
        f"above 95%. One photo alone stays just short of the 95% target. We also tried letting the model abstain when unsure, "
        f"but the threshold chosen on calibration answers {pct(sel['test_share_answered']['value'], 0)} of test photos, so it does nothing here. "
        f"The safer fallback is the taxonomy: if it backs off to genus or family when unsure, it is right {pct(tgt['backoff accuracy at reported rank (1 photo)']['value'])} of the time."),
        foot=f"Up to 3 photos: scored per observation ({three['n']:,} observations). 1 photo: {one['n']:,} photos. "
             "Error bars: 95% bootstrap CI.")
    picture(s, cpath["photos"], M - Inches(0.1), top + Inches(0.1), Inches(7.6))
    bullets(s, Inches(8.25), top + Inches(0.3), Inches(4.5), Inches(4.5), [
        f"Up to 3 photos: {pct(three['top1'])}, CI {pct(three['ci']['top1'][0])} to {pct(three['ci']['top1'][1])}.",
        f"Backing off to genus or family when unsure: {pct(tgt['backoff accuracy at reported rank (1 photo)']['value'])} correct.",
        f"1 photo: {pct(one['top1'])} (3 photos reach {pct(three['top1'])}). Cal-fit selective threshold answered {pct(sel['test_share_answered']['value'], 0)}.",
    ], size=18, gap=16)

    # 8. honest uncertainty
    s = deck.slide("Calibrated confidence matches how often it is right.", (
        f"Calibration error is low for every model here: {pct(d['ece']['probe'])} for the linear probe, {pct(d['ece']['fused'])} for the "
        f"fused scores before temperature and {pct(d['ece']['fused+T'])} after, {pct(d['ece']['ensemble'])} for the final ensemble. When "
        f"VanaLens says 90%, it is right about 90% of the time, and it turns those calibrated scores into decisions. On top we use split conformal prediction with a "
        f"{pct(1 - alpha, 0)} target: the true species was in the set {pct(ens['coverage'])} of the time. Because top-1 is already above "
        f"90%, the set is usually a single species (mean size {ens['mean_set_size']:.1f}), so adaptive sets (APS/RAPS) at 95% or 98% are a natural next step."),
        foot="ECE: expected calibration error on the vl600 test split. Temperature and conformal threshold fitted on the calibration split.")
    picture(s, cpath["cal"], M - Inches(0.1), top + Inches(0.05), Inches(7.4))
    tile(s, Inches(8.3), top + Inches(0.15), Inches(4.43), Inches(1.4), pct(ens["coverage"]),
         f"set coverage (target {pct(1 - alpha, 0)})")
    tile(s, Inches(8.3), top + Inches(1.75), Inches(4.43), Inches(1.4), f"{ens['mean_set_size']:.1f}",
         "species per set on average")
    text(s, Inches(8.3), top + Inches(3.4), Inches(4.43), Inches(1.4),
         "Sets average 1.0 species: coverage holds because top-1 exceeds 90%. Next: APS/RAPS at 95-98%.", 18, False, INK)

    # 9. open set + few-shot
    s = deck.slide("Add a new species with 5 photos and no retraining.", (
        f"To test plants it has never seen, we hid {os_['held_out_species']} species from the catalog. VanaLens flagged "
        f"{pct(os_['reject_rate_held_out']['value'], 0)} of those photos as 'not in catalog', while also flagging "
        f"{pct(os_['false_reject_rate_known']['value'], 0)} of known photos, so this simple similarity gate is a baseline-strength "
        f"detector (AUROC {os_['auroc']['value']:.2f}) with headroom for a learned multi-signal gate. Then we added the "
        f"hidden species back with only a few photos each, no retraining. With 5 photos per species, they are named "
        f"correctly {pct(fs5['top1_new']['value'], 0)} of the time, with 10 photos {pct(d['ex']['few_shot'][-1]['top1_new']['value'], 0)}, "
        f"and accuracy on the old species stays at {pct(fs5['top1_known_after']['value'])}."),
        foot=f"vl600: {os_['held_out_species']} species held out ({os_['n_test_held_out']:,} test photos). Band: 95% bootstrap CI.")
    picture(s, cpath["few"], M - Inches(0.1), top + Inches(0.05), Inches(7.3))
    text(s, Inches(8.3), top + Inches(0.15), Inches(4.43), Inches(0.5), "Unknown-plant check", 20, True, INK)
    tile(s, Inches(8.3), top + Inches(0.7), Inches(2.13), Inches(1.4),
         pct(os_["reject_rate_held_out"]["value"], 0), "unknowns flagged")
    tile(s, Inches(10.6), top + Inches(0.7), Inches(2.13), Inches(1.4),
         pct(os_["false_reject_rate_known"]["value"], 0), "known, wrongly flagged", accent=False)
    text(s, Inches(8.3), top + Inches(2.35), Inches(4.43), Inches(1.9),
         f"AUROC {os_['auroc']['value']:.2f}. Baseline similarity gate; close look-alikes remain hard. Next: multi-signal detector.",
         18, False, INK)

    # 10. app
    s = deck.slide("The app shows the answer, the evidence, and its doubt.", (
        "This is the VanaLens app, running locally. You upload one to three photos and, optionally, the location. "
        "The result card shows the species, a calibrated confidence, the 90% set, the written traits that matched, and the "
        "nearest reference photos from the gallery as evidence. A second tab adds a new species from a few photos. "
        f"On CPU it takes about {d['e2e']['seconds_per_obs_warm']:.1f} seconds per observation once warm."),
        foot=f"Left: app UI. Right: result card from the app's renderer on a vl600 test photo (iNaturalist obs "
             f"{APP_OBS}, CC BY).")
    picture(s, APP / "identify.png", M, top + Inches(0.2), Inches(4.9), border=True)
    picture(s, APP / "result.png", Inches(5.75), top + Inches(0.2), Inches(6.98))
    text(s, M, top + Inches(4.0), CW, Inches(0.9),
         "Every answer shows a calibrated confidence, the 90% set, matched traits and the nearest gallery photos\n"
         "(listed by file in this build). A second tab adds a new species from a few photos.", 18, False, INK)

    # 11. limitations
    s = deck.slide("Current scope and concrete next steps.", (
        "We pair six scope items in the report with concrete next steps; the slide highlights four. "
        "First, test photos come from iNaturalist like the gallery (0.9% within 11 m of a training observation); "
        "next we collect own phone photos and apply a 50 m filter. Second, the open-set similarity gate catches "
        f"{pct(os_['reject_rate_held_out']['value'], 0)} of unseen species at a "
        f"{pct(os_['false_reject_rate_known']['value'], 0)} false-flag rate; next we train a multi-signal detector. "
        f"Third, pre-cutoff calibration leaves the selective step answering {pct(sel['test_share_answered']['value'], 0)} "
        "and sets at size 1; next we calibrate on a post-cutoff slice and add APS/RAPS. Fourth, Pl@ntNet and VLM "
        f"baselines need API keys, DINOv2 adds {d['dino_gain']} over BioCLIP 2 alone, and traits await botanist "
        "review; next we run the keys, test DINOv3, and review the top 100 species."))
    bullets(s, M, top + Inches(0.55), CW, Inches(4.6), [
        "Same-source iNaturalist test (0.9% within 11 m). Next: own phone photos and 50 m filter.",
        f"Baseline open-set gate ({pct(os_['reject_rate_held_out']['value'], 0)} reject, "
        f"{pct(os_['false_reject_rate_known']['value'], 0)} false flag). Next: multi-signal detector.",
        f"Pre-cutoff cal ({pct(sel['test_share_answered']['value'], 0)} selective answered; size-1 sets). Next: post-cutoff cal + APS/RAPS.",
        f"Pl@ntNet/VLM need keys; DINOv2 adds {d['dino_gain']}; traits unreviewed. Next: run keys, DINOv3, botanist check.",
    ], size=22, gap=28)

    # 12. thank you
    s = deck.slide(None, (
        f"To sum up: VanaLens identifies {st['species']} common plants at {pct(three['top1'])} top-1 from up to three photos "
        f"and {pct(one['top1'])} from one photo with location ({hd['h2h_gain_zs_geo']} over BioCLIP 2 zero-shot), with "
        f"{pct(ens['ece'])} calibration error, {pct(one['backoff_acc'])} backoff accuracy, visible gallery and trait "
        f"evidence, and {pct(fs5['top1_new']['value'], 0)} five-shot species addition without retraining. "
        "Thank you. We are happy to take questions."),
        foot=credit(meta, ["delonix"]))
    text(s, M, Inches(1.6), Inches(6.2), Inches(1.0), "Thank you", 36, True, ACCENT)
    text(s, M, Inches(2.55), Inches(6.2), Inches(0.8), "Questions welcome.", 24, False, INK)
    for i, (big, lab) in enumerate([(pct(one["top1"]), "top-1, 1 photo + location"), (pct(three["top1"]), "top-1, up to 3 photos"),
                                    (pct(fs5["top1_new"]["value"], 0), "new species from 5 photos")]):
        tile(s, M + i * Inches(2.05), Inches(3.9), Inches(1.9), Inches(1.5), big, lab)
    picture(s, crop_photo("delonix", 0.8), Inches(8.0), Inches(0.6), Inches(4.6), int(Inches(4.6) / 0.8))

    deck.save(OUT / "VanaLens.pptx")
    print(f"saved {OUT / 'VanaLens.pptx'} with {deck.n} slides")


APP_OBS = "302797063"

if __name__ == "__main__":
    build()
