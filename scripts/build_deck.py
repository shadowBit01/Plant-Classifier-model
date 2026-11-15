"""Lane L7 (pitch). Build pitch/VanaLens.pptx (8 slides, 16:9) and its chart PNGs.

Every number on the slides is read from runs/*.json at build time. A metric with no result file is
shown as "not measured" instead of a value. Run from the repo root:

    python scripts/build_deck.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "pitch"

# Light theme
INK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x5F, 0x6B, 0x76)
ACCENT = RGBColor(0x1B, 0x7F, 0x5B)  # leaf green, used for "ours"
SOFT = RGBColor(0xEE, 0xF5, 0xF1)
WARN = RGBColor(0xA8, 0x5A, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEX_OURS, HEX_BASE, HEX_MISSING = "#1B7F5B", "#9AA5AF", "#D9DEE2"
FONT = "Segoe UI"


# ---------------------------------------------------------------- data
def load(rel: str) -> dict[str, Any] | None:
    p = RUNS / rel
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x: float | None, nd: int = 1) -> str:
    return "not measured" if x is None else f"{100 * x:.{nd}f}%"


def gather() -> dict[str, Any]:
    final = load("L6/final.json") or {}
    abl = load("L3/ablation_inat_india.json") or {}
    b2 = load("L2/b2_inat_india.json") or {}
    e2e = load("L6/e2e_inat_india.json") or {}
    rows = {r["method"]: r for r in final.get("datasets", {}).get("inat_india", [])}
    few_shot = load("L6/few_shot_inat_india.json") or load("L3/few_shot_inat_india.json")
    return {"final": final, "rows": rows, "abl": abl, "b2": b2, "e2e": e2e, "few_shot": few_shot}


def ours_row(d: dict[str, Any]) -> dict[str, Any]:
    """Best complete VanaLens row: the final system as last run (geo prior on if it ran)."""
    for k in ("+ geo prior", "fused+T+sets (rerun)", "fused+T+sets"):
        r = d["rows"].get(k)
        if r and r.get("status") == "ok":
            return r
    return {}


# ---------------------------------------------------------------- charts
def _style(ax: Any) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#C9CFD4")
    ax.tick_params(colors="#3A4550", labelsize=11, length=0)
    ax.grid(axis="x", color="#E6E9EC", linewidth=0.8)
    ax.set_axisbelow(True)


def hbar(path: Path, items: list[tuple[str, float | None, str]], xlabel: str,
         xlim: tuple[float, float], fmt: str = "{:.1f}%", scale: float = 100) -> None:
    """Horizontal bars. items = (label, value or None, color). None draws a 'not run' stub."""
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(items) + 1.0), dpi=200)
    labels = [i[0] for i in items][::-1]
    for y, (_, v, c) in enumerate(items[::-1]):
        if v is None:
            ax.barh(y, xlim[1] - xlim[0], left=xlim[0], color="none", edgecolor=HEX_MISSING,
                    hatch="///", height=0.62, linewidth=0)
            ax.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]), y, "not run", va="center",
                    fontsize=10, color="#5F6B76")
        else:
            ax.barh(y, scale * v - xlim[0], left=xlim[0], color=c, height=0.62)
            ax.text(scale * v + 0.01 * (xlim[1] - xlim[0]), y, fmt.format(scale * v),
                    va="center", fontsize=11, color="#1F2933")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=11, color="#3A4550")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)


def chart_results(d: dict[str, Any]) -> Path:
    r, ours = d["rows"], ours_row(d)
    items: list[tuple[str, float | None, str]] = [("VanaLens (ours)", ours.get("top1"), HEX_OURS)]
    for key, label in (("B2 BioCLIP 2 kNN", "B2 BioCLIP 2 kNN"),
                       ("B3 linear probe", "B3 linear probe"),
                       ("B0 PlantNet API", "B0 Pl@ntNet API"),
                       ("B1 BioCLIP zero-shot", "B1 BioCLIP zero-shot"),
                       ("B4 VLM", "B4 general VLM")):
        row = r.get(key, {})
        items.append((label, row.get("top1") if row.get("status") == "ok" else None, HEX_BASE))
    p = OUT / "results.png"
    hbar(p, items, "Top-1 accuracy (%), inat_india test, post-July-2025", (0, 100))
    return p


def chart_ablation(d: dict[str, Any]) -> Path:
    r = d["rows"]
    steps = [("kNN only", "knn"), ("text traits only", "text"), ("probe only", "probe"),
             ("fused (probe + text)", "fused"), ("+ temp + sets (rerun, new traits)", "fused+T+sets (rerun)"),
             ("+ geo prior", "+ geo prior"), ("+ reranker", "+ reranker")]
    items = []
    for label, key in steps:
        row = r.get(key, {})
        v = row.get("top1") if row.get("status") == "ok" else None
        items.append((label, v, HEX_OURS if key in ("fused", "fused+T+sets (rerun)", "+ geo prior") else HEX_BASE))
    p = OUT / "ablation.png"
    hbar(p, items, "Top-1 accuracy (%), inat_india test", (0, 100))
    return p


def chart_calibration(d: dict[str, Any]) -> Path | None:
    rows = d["abl"].get("rows", {})
    if "fused" not in rows or "fused+T" not in rows:
        return None
    items = [("probe only", rows.get("probe", {}).get("ece"), HEX_BASE),
             ("fused, raw", rows["fused"].get("ece"), HEX_BASE),
             ("fused + temperature", rows["fused+T"].get("ece"), HEX_OURS)]
    p = OUT / "calibration.png"
    hbar(p, items, "Expected calibration error (lower is better)", (0, 0.2), fmt="{:.3f}", scale=1)
    return p


def chart_few_shot(d: dict[str, Any]) -> Path | None:
    fs = d["few_shot"]
    if not fs:
        return None
    pts = sorted((int(k), float(v)) for k, v in fs.get("top1_by_shots", {}).items())
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=200)
    ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], color=HEX_OURS, linewidth=2, marker="o", markersize=8)
    ax.set_xlabel("Reference photos per new species", fontsize=11)
    ax.set_ylabel("Top-1 (%)", fontsize=11)
    _style(ax)
    p = OUT / "few_shot.png"
    fig.tight_layout()
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- slide helpers
def add_text(slide: Any, x: float, y: float, w: float, h: float, text: str | list[str], size: int = 18,
             color: RGBColor = INK, bold: bool = False, align: Any = PP_ALIGN.LEFT, bullets: bool = False) -> Any:
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(8 if bullets else 2)
        run = para.add_run()
        run.text = ("•  " + line) if bullets else line
        run.font.size, run.font.bold, run.font.name = Pt(size), bold, FONT
        run.font.color.rgb = color
    return tb


def new_slide(prs: Any, num: int, kicker: str, title: str, notes: str) -> Any:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), prs.slide_height)
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
    add_text(s, 0.6, 0.35, 10, 0.4, kicker.upper(), 12, ACCENT, bold=True)
    add_text(s, 0.6, 0.7, 12, 0.9, title, 32, INK, bold=True)
    add_text(s, 11.9, 7.0, 1.2, 0.35, f"{num} / 8", 11, MUTED, align=PP_ALIGN.RIGHT)
    add_text(s, 0.6, 7.0, 4, 0.35, "VanaLens", 11, MUTED)
    s.notes_slide.notes_text_frame.text = script_notes().get(num, notes)
    return s


def script_notes() -> dict[int, str]:
    """Speaker notes per slide number, parsed from the '## Slide N' sections of pitch/script.md."""
    p = OUT / "script.md"
    if not p.exists():
        return {}
    out: dict[int, str] = {}
    for block in p.read_text(encoding="utf-8").split("\n## Slide ")[1:]:
        head, _, body = block.partition("\n")
        num = head.split(" ", 1)[0]
        if num.isdigit():
            out[int(num)] = head.split(" ", 1)[1] + "\n" + " ".join(body.split())
    return out


def card(slide: Any, x: float, y: float, w: float, h: float, head: str, body: str,
         fill: RGBColor = SOFT, head_color: RGBColor = ACCENT) -> None:
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.08
    sh.fill.solid(); sh.fill.fore_color.rgb = fill; sh.line.fill.background()
    add_text(slide, x + 0.2, y + 0.15, w - 0.4, 0.6, head, 22, head_color, bold=True)
    add_text(slide, x + 0.2, y + 0.8, w - 0.4, h - 0.9, body, 15, INK)


def picture(slide: Any, path: Path, x: float, y: float, w: float) -> None:
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))


# ---------------------------------------------------------------- deck
def build() -> Path:
    OUT.mkdir(exist_ok=True)
    d = gather()
    ours, r, b2, abl, e2e = ours_row(d), d["rows"], d["b2"], d["abl"], d["e2e"]
    st = abl.get("state", {})
    n_cls = st.get("n_classes")
    n = st.get("n", {})
    fsets = abl.get("rows", {}).get("fused+T+sets", {})
    few = chart_few_shot(d)
    res_png, abl_png, cal_png = chart_results(d), chart_ablation(d), chart_calibration(d)
    b2row = r.get("B2 BioCLIP 2 kNN", {})

    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)  # 13.333 x 7.5 in

    # 1 Problem
    s = new_slide(prs, 1, "The problem", "Plant ID apps answer with false confidence",
                  "See pitch/script.md, slide 1.")
    card(s, 0.6, 2.0, 3.9, 3.8, "One overconfident label",
         "Apps return a single species even when the photo is ambiguous. No 'I am not sure'.")
    card(s, 4.75, 2.0, 3.9, 3.8, "Look-alikes get mixed up",
         "Medicinal herbs and their adulterants look alike. A wrong answer here has real cost.")
    card(s, 8.9, 2.0, 3.9, 3.8, "Local species need retraining",
         "Adding a plant from your own region means collecting data and fine-tuning a model.")

    # 2 Why current approaches fall short
    s = new_slide(prs, 2, "Why current approaches fall short",
                  "Closed sets, no uncertainty, weak open-ended VLMs", "See pitch/script.md, slide 2.")
    tol = b2.get("tol_top1")
    add_text(s, 0.6, 1.9, 7.2, 4.5, [
        "Fine-tuned classifiers are closed-set: a new species means a new training run.",
        "Softmax scores are not calibrated probabilities, so 90% rarely means 90%.",
        f"Open-ended naming is much weaker than choosing from a list: BioCLIP 2 picking from the whole "
        f"tree of life gets {pct(tol)} top-1 on our test split, vs {pct(b2row.get('top1'))} when it "
        f"chooses among our {n_cls} species.",
        "General VLM open-ended baseline (B4): not measured, no LLM key during the build.",
    ], 19, bullets=True)
    card(s, 8.3, 1.9, 4.5, 3.6, "Our answer",
         "Keep strong frozen encoders. Add evidence, calibration and a coverage guarantee on top. Never fine-tune.")

    # 3 System
    s = new_slide(prs, 3, "Our system", "Frozen encoders, three scorers, honest output",
                  "See pitch/script.md, slide 3.")
    steps = [("Photo(s)", "+ GPS from EXIF"), ("Frozen encoders", "BioCLIP 2 + DINO"),
             ("3 scorers", "kNN, probe, text traits"), ("Fuse + calibrate", "log-linear + temperature"),
             ("Geo prior", "p(species | place)"), ("Backoff", "species, genus, family"),
             ("Conformal set", "90% coverage target")]
    w, gap, y = 1.6, 0.15, 2.3
    for i, (h, b) in enumerate(steps):
        x = 0.6 + i * (w + gap)
        sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(1.6))
        sh.fill.solid(); sh.fill.fore_color.rgb = ACCENT if i in (2, 6) else SOFT; sh.line.fill.background()
        c = WHITE if i in (2, 6) else INK
        add_text(s, x + 0.08, y + 0.2, w - 0.16, 0.6, h, 15, c, bold=True, align=PP_ALIGN.CENTER)
        add_text(s, x + 0.08, y + 0.85, w - 0.16, 0.7, b, 11, c, align=PP_ALIGN.CENTER)
    add_text(s, 0.6, 4.4, 12.2, 2.4, [
        "Output: species (or genus when unsure), a calibrated set, the 3 nearest reference photos as evidence, matched traits.",
        "New species: drop photos into the gallery. No retraining, no GPU.",
        "Gated VLM reranker for sets with more than one species: built, but off in this run (no LLM key).",
    ], 17, bullets=True)

    # 4 Demo
    s = new_slide(prs, 4, "Live demo", "Photo in, evidence out", "See pitch/demo.md for exact steps.")
    for i, (h, b) in enumerate([
        ("1  Identify", "Upload a leaf photo. Get the species, a 90% set as chips, and 3 matching reference photos."),
        ("2  Second photo", "Add a flower photo of the same plant. Both photos are fused into one answer and one set."),
        ("3  Add a species", "Name + genus + family + a few photos. Searchable immediately, no retraining."),
    ]):
        card(s, 0.6 + i * 4.15, 2.0, 3.9, 3.4, h, b)
    lat = e2e.get("seconds_per_photo_warm")
    add_text(s, 0.6, 5.7, 12, 0.6,
             f"Runs on a laptop CPU: {lat:.2f} s per photo once warm ({e2e.get('n')} test photos)." if lat
             else "Latency: not measured.", 16, MUTED)

    # 5 Results
    s = new_slide(prs, 5, "Results", f"{pct(ours.get('top1'))} top-1 on photos taken after the model's cutoff",
                  "See pitch/script.md, slide 5.")
    picture(s, res_png, 0.5, 1.75, 7.3)
    add_text(s, 8.1, 1.9, 4.8, 4.6, [
        f"Test: {ours.get('n')} iNaturalist photos, {n_cls} Madhya Pradesh species, all from July 2025 on.",
        f"Macro top-1 {pct(ours.get('macro_top1'))}, top-5 {pct(ours.get('top5'))}.",
        f"Strongest baseline: B2 at {pct(b2row.get('top1'))}.",
        "Pl@ntNet API (B0), BioCLIP zero-shot (B1) and VLM (B4): not run, no API keys.",
        "Caveat: BioCLIP 2 trained on GBIF data; older images may overlap. Hence the post-July-2025 split.",
    ], 15, bullets=True)

    # 6 Ablation
    s = new_slide(prs, 6, "Ablation", "What each piece adds", "See pitch/script.md, slide 6.")
    picture(s, abl_png, 0.5, 1.75, 7.3)
    w_ = st.get("weights", {})
    geo = r.get("+ geo prior", {}); base = r.get("fused+T+sets (rerun)", {})
    add_text(s, 8.1, 1.9, 4.8, 4.8, [
        f"Text traits alone: {pct(r.get('text', {}).get('top1'))} top-1 with zero photos, and they lift the probe "
        f"from {pct(r.get('probe', {}).get('top1'))} to {pct(r.get('fused', {}).get('top1'))}.",
        f"Fitted weights: probe {w_.get('probe')}, text {w_.get('text')}, kNN {w_.get('knn')}. kNN supplies evidence photos, not votes.",
        f"Geo prior: top-1 {pct(base.get('top1'))} to {pct(geo.get('top1'))}; all {n_cls} species are local, so little to remove.",
        "Temperature and sets never change top-1; the rerun gain comes from rebuilt trait prototypes.",
        "Not measured: multi-photo gain, reranker, few-shot curve." if few is None
        else "Not measured: multi-photo gain, reranker.",
    ], 15, bullets=True)

    # 7 Clean test + honest uncertainty
    s = new_slide(prs, 7, "Clean test, honest confidence", "Calibrated confidence on post-cutoff photos",
                  "See pitch/script.md, slide 7.")
    if cal_png:
        picture(s, cal_png, 0.5, 1.9, 7.3)
    add_text(s, 8.1, 1.9, 4.8, 4.8, [
        f"Temperature scaling cuts calibration error from {abl['rows']['fused']['ece']:.3f} to "
        f"{abl['rows']['fused+T']['ece']:.3f}." if cal_png else "Calibration error: not measured.",
        f"Conformal sets, 90% target: {pct(fsets.get('coverage'))} coverage, mean set size {fsets.get('mean_set_size')}. Every set is one species, so coverage equals top-1 here.",
        f"{pct(fsets.get('unknown_rate'))} of photos back off to an 'unknown' rank instead of a forced species name.",
        f"Split sizes: gallery {n.get('gallery')}, calibration {n.get('cal')}, test {n.get('test')}.",
        "Campus photos: not collected, so the campus test was not run.",
    ], 15, bullets=True)

    # 8 Impact
    s = new_slide(prs, 8, "Impact + roadmap", "Trustworthy plant ID for India", "See pitch/script.md, slide 8.")
    card(s, 0.6, 2.0, 3.9, 3.0, "Herb quality checks",
         "Flag medicinal look-alikes with evidence photos a pharmacist can verify.")
    card(s, 4.75, 2.0, 3.9, 3.0, "Biodiversity census",
         "Campus and forest surveys: add local species in minutes, keep honest uncertainty.")
    card(s, 8.9, 2.0, 3.9, 3.0, "Offline mobile",
         "Next: distil to a small ConvNeXt for phones; switch on the VLM reranker; add campus and multi-photo tests.")
    add_text(s, 0.6, 5.5, 12, 0.8,
             f"{pct(ours.get('top1'))} top-1  |  {pct(fsets.get('coverage'))} coverage at a 90% target  |  no fine-tuning",
             22, ACCENT, bold=True, align=PP_ALIGN.CENTER)

    path = OUT / "VanaLens.pptx"
    prs.save(path)
    return path


if __name__ == "__main__":
    p = build()
    prs = Presentation(str(p))  # re-open to prove the file is valid
    print(f"wrote {p.relative_to(ROOT)}: {len(prs.slides)} slides")
