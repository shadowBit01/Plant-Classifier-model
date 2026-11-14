"""Shared paths and constants. FROZEN: ask the human before changing."""
from __future__ import annotations

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("VANALENS_DATA", Path.home() / "vanalens-data"))
CACHE_DIR = Path(os.environ.get("VANALENS_CACHE", Path.home() / "vanalens-cache"))
RUNS_DIR = REPO_DIR / "runs"

BIOCLIP_MODEL = "hf-hub:imageomics/bioclip-2"
DINO_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
BIOCLIP_DIM = 768   # encoders.py must assert this at load time
DINO_DIM = 1024     # encoders.py must assert this at load time
EMB_DIM = BIOCLIP_DIM + DINO_DIM

ALPHA = 0.1          # conformal miscoverage -> 90% prediction sets
GALLERY_CAP = 100    # max reference images per species
BACKOFF_TAU = 0.6    # min probability to report a rank
GBIF_PLANTAE_KEY = 6
LLM_KEY_ENV = "VANALENS_LLM_KEY"
