"""Lane L5 (app). Gradio demo. Develop with VANALENS_MOCK=1; the integration lane switches it to the real model.

Run: VANALENS_MOCK=1 python -m app.app

Requirements (in priority order):
 1 Upload 1-4 photos (JPEG/PNG) of the same plant. Optional lat/lon fields, auto-filled from EXIF
   (vanalens.data.read_exif_gps) when present.
 2 Result card: label at its rank (species or "Genus (species uncertain)"), confidence,
   and the conformal set as chips ("90% sure it's one of these").
 3 Evidence panel: the 3 nearest reference photos (Prediction.evidence) + matched traits.
 4 If Prediction.reranked: show the rationale in a collapsible box.
 5 Hint row: if the user sent one non-flower photo and the set has > 1 species, suggest a flower photo.
 6 "Add a species" tab: name + genus + family + 3-10 photos -> pipeline.add_species(...).
 7 Works on a phone browser (single column below 700 px). No external CSS/CDN needed.
Never call models directly: only vanalens.pipeline.get_pipeline().
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

from vanalens.pipeline import get_pipeline
from vanalens.contracts import Taxon

# Try to import read_exif_gps; it may be NotImplementedError at runtime (dev stub).
try:
    from vanalens.data import read_exif_gps as _read_exif_gps
except Exception:
    _read_exif_gps = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_exif_gps(path: str) -> tuple[float, float] | None:
    """Return (lat, lon) from EXIF or None if unavailable / not implemented."""
    if _read_exif_gps is None:
        return None
    try:
        return _read_exif_gps(path)
    except Exception:
        return None


def _chips_html(pred_set: list[int], taxa_map: dict[int, str]) -> str:
    """Render the conformal prediction set as inline HTML chips."""
    chips = "".join(
        f'<span style="display:inline-block;background:#e8f5e9;border:1px solid #81c784;'
        f'border-radius:16px;padding:2px 10px;margin:2px 3px;font-size:0.85em">'
        f'{taxa_map.get(i, f"class {i}")}</span>'
        for i in pred_set
    )
    return f'<div style="margin-top:6px"><em style="color:#555">90% sure it\'s one of these:</em><br>{chips}</div>'


def _result_html(pred, pipeline) -> str:
    """Build the full result card HTML from a Prediction."""
    taxa_map = {t.idx: t.species for t in pipeline.taxa} if hasattr(pipeline, "taxa") else {}

    # Rank label
    if pred.rank == "species":
        label_html = f'<span style="font-size:1.3em;font-weight:700">{pred.label}</span>'
    else:
        label_html = (
            f'<span style="font-size:1.3em;font-weight:700">{pred.label}</span>'
            f' <span style="color:#888;font-size:0.9em">(species uncertain)</span>'
        )

    conf_pct = f"{pred.confidence * 100:.1f}%"

    chips = _chips_html(pred.pred_set, taxa_map) if pred.pred_set else ""

    # Traits
    traits_html = ""
    if pred.traits:
        trait_items = "".join(f"<li>{t}</li>" for t in pred.traits)
        traits_html = f"<ul style='margin:4px 0 0 16px;padding:0'>{trait_items}</ul>"

    # Evidence
    evidence_html = ""
    if pred.evidence:
        ev_spans = "".join(
            f'<span style="display:inline-block;background:#f5f5f5;border:1px solid #ddd;'
            f'border-radius:4px;padding:2px 7px;margin:2px 2px;font-size:0.8em;font-family:monospace">'
            f'{e}</span>'
            for e in pred.evidence[:3]
        )
        evidence_html = (
            f'<div style="margin-top:8px"><strong>Nearest references:</strong><br>{ev_spans}</div>'
        )

    # Rationale (collapsible)
    rationale_html = ""
    if pred.reranked and pred.rationale:
        rationale_html = (
            f'<details style="margin-top:8px"><summary style="cursor:pointer;color:#1565c0">'
            f'🔍 Reranking rationale</summary>'
            f'<p style="margin:4px 8px;color:#333">{pred.rationale}</p></details>'
        )

    return f"""
<div style="
  border:2px solid #81c784;border-radius:12px;padding:16px;
  background:#f9fbe7;margin:8px 0
">
  <div style="margin-bottom:4px">{label_html}</div>
  <div style="color:#2e7d32;font-weight:600">Confidence: {conf_pct}</div>
  {chips}
  {traits_html}
  {evidence_html}
  {rationale_html}
</div>
""".strip()


def _hint_html() -> str:
    return (
        '<div style="background:#fff3e0;border:1px solid #ffb300;border-radius:8px;'
        'padding:10px;margin-top:8px">'
        '📸 <strong>Tip:</strong> Adding a flower photo often narrows the species identification.'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Predict callback
# ---------------------------------------------------------------------------

def predict_fn(images, lat_str: str, lon_str: str):
    """Core prediction callback. images is a list of file paths from gr.File."""
    if not images:
        return "<p style='color:#c62828'>Please upload at least one photo.</p>", ""

    # Build path list (gr.File returns filepath strings directly in Gradio 4+)
    paths = []
    for img in images:
        if img is None:
            continue
        if isinstance(img, str):
            paths.append(img)
        elif hasattr(img, "name"):
            paths.append(img.name)
        else:
            paths.append(str(img))

    if not paths:
        return "<p style='color:#c62828'>No valid image files found.</p>", ""

    # Attempt EXIF GPS from first image if lat/lon not provided
    lat, lon = None, None
    if lat_str.strip():
        try:
            lat = float(lat_str.strip())
        except ValueError:
            pass
    if lon_str.strip():
        try:
            lon = float(lon_str.strip())
        except ValueError:
            pass

    if lat is None and lon is None:
        gps = _try_exif_gps(paths[0])
        if gps:
            lat, lon = gps

    # Run prediction
    pipeline = get_pipeline()
    try:
        pred = pipeline.predict(paths, lat=lat, lon=lon)
    except Exception as exc:
        return f"<p style='color:#c62828'>Prediction error: {exc}</p>", ""

    result_html = _result_html(pred, pipeline)

    # Hint: one non-flower photo + multiple species in pred_set
    hint_html = ""
    if len(paths) == 1 and len(pred.pred_set) > 1:
        organ = getattr(pred, "organ", None)
        if organ and organ != "flower":
            hint_html = _hint_html()
        elif organ is None:
            hint_html = _hint_html()

    return result_html, hint_html


# ---------------------------------------------------------------------------
# Add-species callback
# ---------------------------------------------------------------------------

def add_species_fn(name: str, genus: str, family: str, photos):
    if not name.strip():
        return "⚠️ Species name is required."
    if not photos:
        return "⚠️ Please upload 3–10 reference photos."

    paths = []
    for img in photos:
        if img is None:
            continue
        if isinstance(img, str):
            paths.append(img)
        elif hasattr(img, "name"):
            paths.append(img.name)
        else:
            paths.append(str(img))

    if len(paths) < 3:
        return f"⚠️ Please upload at least 3 photos (got {len(paths)})."
    if len(paths) > 10:
        return "⚠️ Please upload at most 10 photos."

    pipeline = get_pipeline()
    taxon = Taxon(idx=-1, species=name.strip(), genus=genus.strip(), family=family.strip())
    try:
        pipeline.add_species(taxon, paths)
    except Exception as exc:
        return f"❌ Error: {exc}"

    return f"✅ Added **{name.strip()}** to the pipeline."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

MOBILE_CSS = """
/* Mobile-friendly single-column layout below 700 px. No external CDN needed. */
body { font-family: system-ui, sans-serif; }

@media (max-width: 700px) {
  .gradio-container { padding: 8px !important; }
  .gr-row { flex-direction: column !important; }
  .gr-column { min-width: 100% !important; max-width: 100% !important; }
  .upload-container { min-height: 160px !important; }
  #result-card, #hint-row { font-size: 0.95em; }
}

#result-card { min-height: 60px; }
"""


def build_ui() -> gr.Blocks:
    pipeline = get_pipeline()

    with gr.Blocks(title="VanaLens \u2013 Plant Identifier") as demo:
        gr.Markdown(
            "# 🌿 VanaLens – Plant Identifier\n"
            "Upload 1–4 photos of the **same plant** (JPEG/PNG) for instant identification."
        )

        with gr.Tabs():
            # ----------------------------------------------------------------
            # Tab 1: Identify
            # ----------------------------------------------------------------
            with gr.Tab("Identify"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_upload = gr.File(
                            label="Plant photos (1–4, JPEG/PNG)",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        with gr.Row():
                            lat_field = gr.Textbox(
                                label="Latitude (optional)",
                                placeholder="e.g. 12.9716",
                                scale=1,
                            )
                            lon_field = gr.Textbox(
                                label="Longitude (optional)",
                                placeholder="e.g. 77.5946",
                                scale=1,
                            )
                        identify_btn = gr.Button("🔍 Identify", variant="primary")

                    with gr.Column(scale=1):
                        result_html = gr.HTML(elem_id="result-card", label="Result")
                        hint_html = gr.HTML(elem_id="hint-row")

                identify_btn.click(
                    fn=predict_fn,
                    inputs=[image_upload, lat_field, lon_field],
                    outputs=[result_html, hint_html],
                )

                # Auto-fill GPS from EXIF when first image is uploaded
                def autofill_gps(files):
                    if not files:
                        return "", ""
                    path = files[0] if isinstance(files[0], str) else getattr(files[0], "name", None)
                    if path:
                        gps = _try_exif_gps(path)
                        if gps:
                            return str(gps[0]), str(gps[1])
                    return "", ""

                image_upload.change(
                    fn=autofill_gps,
                    inputs=[image_upload],
                    outputs=[lat_field, lon_field],
                )

            # ----------------------------------------------------------------
            # Tab 2: Add a species
            # ----------------------------------------------------------------
            with gr.Tab("Add a species"):
                gr.Markdown(
                    "Add a new species to the pipeline with 3–10 reference photos.\n"
                    "*(No retraining – few-shot kNN addition.)*"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        sp_name = gr.Textbox(label="Species name", placeholder="e.g. Tectona grandis")
                        sp_genus = gr.Textbox(label="Genus", placeholder="e.g. Tectona")
                        sp_family = gr.Textbox(label="Family", placeholder="e.g. Lamiaceae")
                        sp_photos = gr.File(
                            label="Reference photos (3–10)",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        add_btn = gr.Button("➕ Add species", variant="primary")
                    with gr.Column(scale=1):
                        add_status = gr.Markdown("")

                add_btn.click(
                    fn=add_species_fn,
                    inputs=[sp_name, sp_genus, sp_family, sp_photos],
                    outputs=[add_status],
                )

    return demo


if __name__ == "__main__":
    build_ui().launch(css=MOBILE_CSS)
