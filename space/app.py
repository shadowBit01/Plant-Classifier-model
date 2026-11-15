"""Hugging Face Space entry point for VanaLens (app/app.py).

Defaults VANALENS_MOCK=0 and loads state from the cache path given by env
(VANALENS_CACHE / VANALENS_DATA / VANALENS_STATE / VANALENS_DATASET).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "vanalens").is_dir() and (ROOT.parent / "vanalens").is_dir():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def configure_space_env() -> tuple[Path, Path, str]:
    """Configure environment defaults for Hugging Face Spaces and sync vanalens.config."""
    os.environ.setdefault("VANALENS_MOCK", "0")
    has_explicit_data = "VANALENS_DATA" in os.environ
    default_cache = ROOT / "cache" if (ROOT / "cache").exists() else Path.home() / "vanalens-cache"
    default_data = ROOT / "data" if (ROOT / "data").exists() else Path.home() / "vanalens-data"
    os.environ.setdefault("VANALENS_CACHE", str(default_cache))
    os.environ.setdefault("VANALENS_DATASET", "inat_india")

    from vanalens import config as vanalens_config

    dataset = os.environ["VANALENS_DATASET"]
    state_env = os.environ.get("VANALENS_STATE")
    if state_env:
        sp = Path(state_env)
        if sp.suffix == ".json":
            dataset = sp.stem
            os.environ["VANALENS_DATASET"] = dataset
            if sp.parent.name == "state":
                os.environ["VANALENS_CACHE"] = str(sp.parent.parent)
        elif sp.is_dir():
            os.environ["VANALENS_CACHE"] = str(sp)

    cache_dir = Path(os.environ["VANALENS_CACHE"])
    if has_explicit_data:
        data_dir = Path(os.environ["VANALENS_DATA"])
    elif (cache_dir / dataset / "labels.json").exists():
        data_dir = cache_dir
    else:
        data_dir = default_data

    vanalens_config.CACHE_DIR = cache_dir
    vanalens_config.DATA_DIR = data_dir
    return cache_dir, data_dir, dataset


configure_space_env()

from app.app import MOBILE_CSS, build_ui, load_pipeline  # noqa: E402


def build_space_ui():
    """Sync environment variables and build the Gradio Blocks UI."""
    configure_space_env()
    return build_ui()


if __name__ == "__main__":
    build_space_ui().launch(css=MOBILE_CSS)
