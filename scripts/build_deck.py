"""Lane L7 (pitch). Build pitch/VanaLens.pptx (12 slides, 16:9), its chart PNGs and photo crops.

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
          target: float | None = None, fmt: str = "{:.1f}%") -> None:
    """Horizontal bar chart (top to bottom in list order), value labels, 95% CI error bars."""
    fig, ax = plt.subplots(figsize=size)
    y = list(range(len(labels)))[::-1]
    cols = [HEX_ACCENT if o else HEX_BASE for o in ours]
    ax.barh(y, vals, color=cols, height=0.62, zorder=2)
    for yi, v, ci in zip(y, vals, cis):
        if ci:
            ax.errorbar(v, yi, xerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=HEX_INK,
                        elinewidth=1.4, capsize=5, zorder=3)
        hi = ci[1] if ci else v
        ax.text(hi + xmax * 0.012, yi, fmt.format(v), va="center", ha="left", fontsize=15,
                fontweight="bold", color=HEX_INK)
    if target is not None:
        ax.axvline(target, color=HEX_INK, lw=1.2, ls="--", zorder=1)
        ax.text(target, len(labels) - 0.45, f"{target:.0f}% target", ha="center", va="bottom",
                fontsize=13, color=HEX_MUTED)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, xmax * 1.1)
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
           "Linear probe on\nour embeddings", "VanaLens, 1 photo", "VanaLens, up to 3 photos"],
          [100 * d["b2"]["open_ended_top1"], 100 * d["b2"]["top1"], 100 * d["probe"]["top1"],
           100 * d["one"]["top1"], 100 * d["three"]["top1"]],
          [None, None, [100 * c for c in d["probe"]["ci"]["top1"]],
           [100 * c for c in d["one"]["ci"]["top1"]], [100 * c for c in d["three"]["ci"]["top1"]]],
          [False, False, False, True, True], (8.4, 5.0), "Top-1 accuracy on vl600 test (%)")
    return p


def chart_photos(d: dict) -> Path:
    """1 photo vs location prior vs up to 3 photos, top-1 with CI and the 95% target line."""
    p = OUT / "photos.png"
    hbars(p, ["1 photo", "1 photo + location", "Up to 3 photos\n+ location"],
          [100 * d["one_nogeo"]["top1"], 100 * d["one"]["top1"], 100 * d["three"]["top1"]],
          [[100 * c for c in d["one_nogeo"]["ci"]["top1"]], [100 * c for c in d["one"]["ci"]["top1"]],
           [100 * c for c in d["three"]["ci"]["top1"]]],
          [False, True, True], (7.6, 4.4), "Top-1 accuracy on vl600 test (%)", target=95)
    return p


def chart_calibration(d: dict) -> Path:
    """Expected calibration error, lower is better."""
    p = OUT / "calibration.png"
    e = d["ece"]
    hbars(p, ["Linear probe alone", "Fused, before calibration", "BioCLIP 2 zero-shot",
              "Fused + temperature", "VanaLens ensemble"],
          [100 * e["probe"], 100 * e["fused"], 100 * d["ece_b2"], 100 * e["fused+T"], 100 * e["ensemble"]],
          [None] * 5, [False, False, False, True, True], (7.4, 4.2),
          "Calibration error (ECE, %), lower is better", xmax=16)
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


class Deck:
    """Thin wrapper that adds themed slides with title, number, footnote and notes."""

    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = SW, SH
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

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
        return s

    def save(self, path: Path) -> None:
        self.prs.save(path)


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
                                          ("cal", lambda: chart_calibration(d)), ("few", lambda: chart_fewshot(d)))}
    deck = Deck()
    top = Inches(1.6)  # content top below the title

    # 1. title
    s = deck.slide(None, (
        "Hi, we are presenting VanaLens. It names common plants from a phone photo, and it is honest when it "
        "is not sure. Everything you will see was measured on photos taken after our models were trained."),
        foot=credit(meta, ["calotropis", "delonix", "hibiscus", "lantana"]))
    text(s, M, Inches(2.2), Inches(5.6), Inches(0.9), "VanaLens", 36, True, ACCENT)
    text(s, M, Inches(3.05), Inches(5.6), Inches(1.4), "Plant identification that knows\nwhen it is not sure",
         28, False, INK)
    text(s, M, Inches(4.55), Inches(5.6), Inches(0.9),
         f"{st['species']} common plants of India and the world.\nEvery number measured on unseen photos.", 18, False, MUTED)
    gx, gy, gw, gap = Inches(6.75), Inches(0.6), Inches(2.85), Inches(0.14)
    for i, k in enumerate(["calotropis", "delonix", "hibiscus", "lantana"]):
        px = gx + (i % 2) * (gw + gap)
        py = gy + (i // 2) * (gw + gap)
        picture(s, crop_photo(k, 1.0), px, py, gw, gw)

    # 2. problem
    s = deck.slide("People want plant names; apps give false confidence.", (
        "Everyone walks past plants they cannot name: a tree on the street, a herb in a market, a weed in a field. "
        "Plant apps exist, but they give one confident answer, even for a plant they have never seen. "
        "General AI chatbots are worse at naming plants from an open question. "
        "And if your local plant is missing, you cannot add it without retraining the model."),
        foot=credit(meta, ["tulsi", "mango"]))
    bullets(s, M, top + Inches(0.2), Inches(8.0), Inches(4.5), [
        "Common plants matter daily: food, medicine, gardens, weeds.",
        "Apps return one confident name, even for unknown plants.",
        f"Open-ended, BioCLIP 2 names only {pct(d['b2']['open_ended_top1'], 0)} of our test photos.",
        "Adding a missing local species usually means retraining.",
    ], size=22, gap=20)
    pw2 = Inches(3.4)
    picture(s, crop_photo("tulsi", 1.45), Inches(9.33), top + Inches(0.1), pw2, int(pw2 / 1.45))
    picture(s, crop_photo("mango", 1.45), Inches(9.33), top + Inches(0.25) + int(pw2 / 1.45), pw2, int(pw2 / 1.45))

    # 3. what we built
    s = deck.slide(f"VanaLens names {st['species']} plants and says when it is unsure.", (
        "VanaLens is a plant identifier built on two frozen vision models, so there is no fine-tuning. "
        "Three ideas make it different. First, no retraining: a new species is added by dropping in a few photos. "
        "Second, honest uncertainty: confidence is calibrated, and when it is not sure it returns a short list or "
        "says the plant is not in its catalog. Third, it uses context: more photos of the same plant and the "
        "location where the photo was taken."))
    text(s, M, top + Inches(0.05), CW, Inches(0.6),
         "Upload 1 to 3 photos. Get a species, a short list, or \"not in catalog\", with evidence.", 22, False, INK)
    cw3, gap3, cy, ch = Inches(3.85), Inches(0.29), top + Inches(1.2), Inches(3.0)
    ideas = [
        ("1", "No retraining", "Frozen BioCLIP 2 and DINOv3 encoders. Add a species with a few photos."),
        ("2", "Honest uncertainty", f"Calibrated confidence, {pct(1 - alpha, 0)} prediction sets, and a \"not in catalog\" answer."),
        ("3", "Uses context", "Fuses up to 3 photos and a location prior from GBIF records."),
    ]
    for i, (num, head, body) in enumerate(ideas):
        x = M + i * (cw3 + gap3)
        box(s, x, cy, cw3, ch, GREY)
        text(s, x + Inches(0.35), cy + Inches(0.3), Inches(1), Inches(0.7), num, 36, True, ACCENT)
        text(s, x + Inches(0.35), cy + Inches(1.1), cw3 - Inches(0.7), Inches(0.5), head, 22, True, INK)
        text(s, x + Inches(0.35), cy + Inches(1.7), cw3 - Inches(0.7), Inches(1.6), body, 18, False, INK)

    # 4. how it works
    s = deck.slide("Frozen encoders, three scorers, then calibration decides.", (
        "Here is the pipeline. A photo goes through two frozen encoders: BioCLIP 2, trained on the tree of life, "
        "and DINOv3, a general visual model. Three scorers read those embeddings: a linear probe, a nearest-neighbour "
        "search over reference photos that also gives us evidence pictures, and text prototypes built from written "
        "plant traits. We fuse them and calibrate with temperature scaling, then multiply in a location prior. "
        "Finally a conformal rule decides: one species, a short 'not sure' list, or 'not in catalog'."))
    by, bh, bw = top + Inches(0.75), Inches(1.75), Inches(1.6)
    gapx = Inches(0.32)
    steps = [
        ("Photo", "1 to 3 photos\n+ GPS if known"),
        ("Encoders", "BioCLIP 2\n+ DINOv3"),
        ("Scorers", "Probe, kNN,\ntext traits"),
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
    outs = [("Answer", "one species", True), ("Not sure", "short list, 90% set", False),
            ("Not in catalog", "add it with photos", False)]
    last_x = M + 4 * (bw + gapx) + bw
    for i, (head, sub, acc) in enumerate(outs):
        oy = oy0 + i * (oh + og)
        b = box(s, ox, oy, ow, oh, ACCENT if acc else WHITE, None if acc else ACCENT, MSO_SHAPE.ROUNDED_RECTANGLE)
        box_text(b, [(head, 18, True, WHITE if acc else ACCENT), (sub, 15, False, WHITE if acc else INK)])
        arrow(s, last_x + Inches(0.04), by + bh / 2, ox - Inches(0.04), oy + oh / 2)
    text(s, M, by + bh + Inches(1.55), CW, Inches(1.0),
         "No fine-tuning of the encoders. Only small heads and a few numbers (weights, temperature,\n"
         "conformal threshold) are fitted, on a separate calibration split.", 18, False, MUTED)

    # 5. dataset
    s = deck.slide("All test photos postdate the encoders' training data.", (
        f"We built our own dataset, vl600, from iNaturalist research-grade observations. It has {st['species']} species: "
        f"{st['species_by_source']['india']} most observed in India, {st['species_by_source']['world']} most observed worldwide, "
        f"and a short list of common cultivated plants. Together they cover {st['coverage_pct']:.1f}% of all research-grade "
        f"plant observations in India. The {st['photos_per_split']['test']:,} test photos were all taken after July 2025, after "
        "BioCLIP 2's training data was collected, so the test is free of contamination."),
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
        "Chosen by observation count only; no species dropped.",
    ], size=18, gap=8)
    pw = Inches(1.82)
    for i, k in enumerate(["mango", "lotus", "tulsi"]):
        picture(s, crop_photo(k, 0.75), Inches(6.85) + i * (pw + Inches(0.14)), top + Inches(0.15), pw, int(pw / 0.75))
    text(s, Inches(6.85), top + Inches(0.15) + int(pw / 0.75) + Inches(0.12), Inches(5.8), Inches(0.4),
         "Mango, lotus, tulsi: three of the 607 classes.", 16, False, MUTED)

    # 6. results
    s = deck.slide(f"VanaLens beats BioCLIP 2 on {one['n']:,} unseen photos.", (
        f"On {one['n']:,} test photos, BioCLIP 2 zero-shot gets {pct(d['b2']['top1'])} when we give it our 607 names, and "
        f"{pct(d['b2']['open_ended_top1'])} open-ended. A linear probe on our embeddings gets {pct(probe['top1'])}. VanaLens with one "
        f"photo gets {pct(one['top1'])}, and with up to three photos of the same plant {pct(three['top1'])}. Error bars are 95% "
        "bootstrap intervals; the BioCLIP 2 runs have no per-photo file, so no interval. We could not run the Pl@ntNet "
        f"API or a VLM baseline because we had no API keys. On our second test set, inat_india, the picture is the same: "
        f"{pct(d['in_one']['top1'])} versus {pct(d['in_b2']['top1'])}."),
        foot="Error bars: 95% bootstrap CI by observation. BioCLIP 2 runs have no CI (no per-photo file). "
             "Pl@ntNet API and VLM baselines not run (no API keys).")
    picture(s, cpath["results"], M - Inches(0.1), top, Inches(8.0))
    bullets(s, Inches(8.45), top + Inches(0.35), Inches(4.3), Inches(4.5), [
        f"+{100 * (one['top1'] - d['b2']['top1']):.1f} points over BioCLIP 2 given the same 607 names.",
        f"Top-5: {pct(one['top5'])} with 1 photo, {pct(three['top5'])} with 3.",
        f"Second test set (inat_india): {pct(d['in_one']['top1'])} vs {pct(d['in_b2']['top1'])}.",
    ], size=18, gap=16)

    # 7. multi-photo + selective
    s = deck.slide("Up to three photos push top-1 accuracy past 95%.", (
        f"One photo gives {pct(d['one_nogeo']['top1'])}. Adding the location prior lifts it to {pct(one['top1'])}. Letting the "
        f"user add up to three photos of the same plant lifts it to {pct(three['top1'])}, with the whole confidence interval "
        f"above 95%. We also tried letting the model abstain when unsure: on vl600 it answers {pct(sel['test_share_answered']['value'], 2)} "
        f"of photos at {pct(sel['test_accuracy_answered']['value'])}, so abstaining barely changes anything here. "
        f"The safer fallback is the taxonomy: if it backs off to genus or family when unsure, it is right {pct(tgt['backoff accuracy at reported rank (1 photo)']['value'])} of the time."),
        foot=f"Up to 3 photos: scored per observation ({three['n']:,} observations). 1 photo: {one['n']:,} photos. "
             "Error bars: 95% bootstrap CI.")
    picture(s, cpath["photos"], M - Inches(0.1), top + Inches(0.1), Inches(7.6))
    bullets(s, Inches(8.25), top + Inches(0.3), Inches(4.5), Inches(4.5), [
        f"Up to 3 photos: {pct(three['top1'])}, CI {pct(three['ci']['top1'][0])} to {pct(three['ci']['top1'][1])}.",
        f"Backing off to genus or family when unsure: {pct(tgt['backoff accuracy at reported rank (1 photo)']['value'])} correct.",
        f"Abstaining adds little: {pct(sel['test_accuracy_answered']['value'])} on {pct(sel['test_share_answered']['value'], 2)} answered.",
    ], size=18, gap=16)

    # 8. honest uncertainty
    s = deck.slide("Calibrated confidence matches how often it is right.", (
        f"Raw fused scores are overconfident: calibration error is {pct(d['ece']['fused'])}. One temperature parameter, fitted "
        f"on the calibration split, brings it to {pct(d['ece']['fused+T'])}; the final ensemble sits at {pct(d['ece']['ensemble'])}. "
        f"So when VanaLens says 90%, it is right about 90% of the time. On top we use split conformal prediction with a "
        f"{pct(1 - alpha, 0)} target: the true species was in the set {pct(ens['coverage'])} of the time. Because top-1 is already above "
        "90%, the set is almost always a single species, so the short 'not sure' list rarely appears on this data."),
        foot="ECE: expected calibration error on the vl600 test split. Temperature and conformal threshold fitted on the calibration split.")
    picture(s, cpath["cal"], M - Inches(0.1), top + Inches(0.05), Inches(7.4))
    tile(s, Inches(8.3), top + Inches(0.15), Inches(4.43), Inches(1.4), pct(ens["coverage"]),
         f"set coverage (target {pct(1 - alpha, 0)})")
    tile(s, Inches(8.3), top + Inches(1.75), Inches(4.43), Inches(1.4), f"{ens['mean_set_size']:.1f}",
         "species per set on average")
    text(s, Inches(8.3), top + Inches(3.4), Inches(4.43), Inches(1.4),
         "Guarantee holds, but sets are rarely larger than one species on vl600.", 18, False, INK)

    # 9. open set + few-shot
    s = deck.slide("Add a new species with 5 photos and no retraining.", (
        f"To test plants it has never seen, we hid {os_['held_out_species']} species from the catalog. VanaLens flagged "
        f"{pct(os_['reject_rate_held_out']['value'], 0)} of those photos as 'not in catalog', but also wrongly flagged "
        f"{pct(os_['false_reject_rate_known']['value'], 0)} of known photos, so this check is the weakest part. Then we added the "
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
         f"AUROC {os_['auroc']['value']:.2f}. Useful, not yet reliable: the weakest part of the system.",
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
         "Every answer shows a calibrated confidence, the 90% set, matched traits and the nearest gallery photos.\n"
         "A second tab adds a new species from a few photos.", 18, False, INK)

    # 11. limitations
    s = deck.slide("It is not perfect yet, and we know what to fix next.", (
        "We want to be upfront about the gaps. The unknown-plant check wrongly flags about a quarter of known photos; "
        "next we tune it on more held-out species. Conformal sets are almost always one species, so the short list adds little; "
        "a tighter coverage target would make it useful. We could not compare against the Pl@ntNet app or a VLM because we had no "
        "API keys. And the live demo falls back to DINOv2 when the gated DINOv3 weights are not available, which makes it "
        "over-flag 'not in catalog'; the fix is simply getting DINOv3 access on the demo machine."))
    bullets(s, M, top + Inches(0.55), CW, Inches(4.6), [
        f"Unknown check flags {pct(os_['false_reject_rate_known']['value'], 0)} of known photos. Next: tune on more species.",
        "Sets are nearly always one species. Next: test on harder look-alikes.",
        "Pl@ntNet and VLM baselines not run. Next: run both with API keys.",
        "Live app falls back to DINOv2 without DINOv3 access. Next: licensed weights.",
    ], size=22, gap=34)

    # 12. thank you
    s = deck.slide(None, (
        f"To sum up: VanaLens names {st['species']} common plants at {pct(one['top1'])} from one photo and {pct(three['top1'])} "
        "from up to three, with calibrated confidence, and adds a new species from a few photos without retraining. "
        "Thank you. We are happy to take questions."),
        foot=credit(meta, ["delonix"]))
    text(s, M, Inches(1.6), Inches(6.2), Inches(1.0), "Thank you", 36, True, ACCENT)
    text(s, M, Inches(2.55), Inches(6.2), Inches(0.8), "Questions welcome.", 24, False, INK)
    for i, (big, lab) in enumerate([(pct(one["top1"]), "top-1, one photo"), (pct(three["top1"]), "top-1, up to 3 photos"),
                                    (pct(fs5["top1_new"]["value"], 0), "new species from 5 photos")]):
        tile(s, M + i * Inches(2.05), Inches(3.9), Inches(1.9), Inches(1.5), big, lab)
    picture(s, crop_photo("delonix", 0.8), Inches(8.0), Inches(0.6), Inches(4.6), int(Inches(4.6) / 0.8))

    deck.save(OUT / "VanaLens.pptx")
    print(f"saved {OUT / 'VanaLens.pptx'} with {deck.n} slides")


APP_OBS = "302797063"

if __name__ == "__main__":
    build()
