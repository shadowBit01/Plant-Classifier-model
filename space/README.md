---
title: VanaLens
emoji: 🌿
colorFrom: green
colorTo: lime
sdk: gradio
sdk_version: "6.32.0"
app_file: app.py
pinned: false
license: mit
short_description: Evidence-backed, uncertainty-aware plant species identifier
---

# VanaLens

Evidence-backed, uncertainty-aware plant species identifier built with frozen vision encoders (BioCLIP 2 + DINOv3), kNN/probe/text-trait fusion, GBIF geographic priors, hierarchical taxonomic backoff, split-conformal prediction sets (90% target coverage), and gated VLM reranking.

## Environment Variables

- `VANALENS_MOCK`: defaults to `0` (real `VanaLens` pipeline). Set to `1` only for mock UI testing.
- `VANALENS_CACHE`: directory containing `state/<dataset>.json`, `emb/<dataset>_gallery.npz`, and optional `text/<dataset>.npz` (defaults to `./cache` in the Space root or `~/vanalens-cache`).
- `VANALENS_DATA`: directory containing `<dataset>/labels.json` (defaults to `./data` in the Space root, falls back to `VANALENS_CACHE` if `<dataset>/labels.json` is stored there, or `~/vanalens-data`).
- `VANALENS_DATASET`: dataset name to load from cache (defaults to `inat_india`).
- `VANALENS_LLM_KEY`: optional Anthropic API key for the gated VLM reranker on multi-species conformal sets.

## Manual Deployment Commands (for Human)

Run from the repository root after authenticating with `huggingface-cli login`:

```bash
export HF_SPACE_REPO="<your-hf-username>/vanalens"
huggingface-cli repo create vana-lens --type space --space_sdk gradio

git clone "https://huggingface.co/spaces/$HF_SPACE_REPO" /tmp/vanalens-space
cp space/README.md space/requirements.txt space/app.py /tmp/vanalens-space/
cp -r vanalens app /tmp/vanalens-space/

mkdir -p /tmp/vanalens-space/cache/state /tmp/vanalens-space/cache/emb /tmp/vanalens-space/cache/text /tmp/vanalens-space/data/inat_india
cp "$VANALENS_CACHE/state/inat_india.json" /tmp/vanalens-space/cache/state/
cp "$VANALENS_CACHE/emb/inat_india_gallery.npz" /tmp/vanalens-space/cache/emb/
cp "$VANALENS_CACHE/text/inat_india.npz" /tmp/vanalens-space/cache/text/ 2>/dev/null || true
cp "$VANALENS_DATA/inat_india/labels.json" /tmp/vanalens-space/data/inat_india/

cd /tmp/vanalens-space
git lfs install
git lfs track "*.npz"
git add .gitattributes README.md requirements.txt app.py vanalens app cache data
git commit -m "Deploy VanaLens Space"
git push
```
