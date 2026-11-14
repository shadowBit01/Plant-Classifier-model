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


def build_ui():
    raise NotImplementedError


if __name__ == "__main__":
    build_ui().launch()
