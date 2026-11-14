"""Lane L4 (context). LLM-written visual traits -> text prototypes. Tests: tests/test_context.py

LLM key: read os.environ[config.LLM_KEY_ENV] ("VANALENS_LLM_KEY") and pass it to the client explicitly.
Never rely on ANTHROPIC_API_KEY (it would switch Claude Code to API billing for the whole team).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .contracts import Taxon

TRAIT_PROMPT = (
    "For each plant species below, list {n} short visual traits (noun phrases, max 8 words each) that help "
    "identify it from a photo of its {organ}: leaf shape, margin, venation, arrangement, surface/colour, "
    "flower colour/structure, fruit, bark. Only traits visible in a photo. Reply as JSON: "
    '{{"<species>": ["trait", ...], ...}}\n\nSpecies:\n{species_list}'
)


def generate_traits(taxa: list[Taxon], organ: str = "leaf or flower", n: int = 6,
                    cache_path: Path | None = None, batch: int = 25, client=None) -> dict[str, list[str]]:
    """Ask the LLM in batches of `batch` species, parse JSON, merge, cache to `cache_path`
    (default CACHE_DIR/traits.json). Re-runs only species missing from the cache. Retry malformed JSON once."""
    raise NotImplementedError


def trait_prompts(species: str, traits: list[str]) -> list[str]:
    """[f"a photo of {species}, which has {t}" for t in traits] + [f"a photo of {species}"] (name prompt last)."""
    raise NotImplementedError


def build_text_protos(taxa: list[Taxon], traits: dict[str, list[str]],
                      encode_text: Callable[[list[str]], np.ndarray]) -> tuple[list[np.ndarray], list[list[str]]]:
    """For each taxon (in idx order): prompts = trait_prompts(...), protos[c] = L2-normalised encode_text(prompts).
    Species with no traits get only the name prompt. Returns (protos, prompt_strings) for scorers.TextScorer."""
    raise NotImplementedError
