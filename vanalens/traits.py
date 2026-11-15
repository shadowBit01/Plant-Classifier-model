"""Lane L4 (context). LLM-written visual traits -> text prototypes. Tests: tests/test_context.py

LLM key: read os.environ[config.LLM_KEY_ENV] ("VANALENS_LLM_KEY") and pass it to the client explicitly.
Never rely on ANTHROPIC_API_KEY (it would switch Claude Code to API billing for the whole team).
"""
from __future__ import annotations

import json
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
    import os
    import anthropic

    from .config import CACHE_DIR, LLM_KEY_ENV

    if cache_path is None:
        cache_path = CACHE_DIR / "traits.json"

    cache_path = Path(cache_path)
    result: dict[str, list[str]] = {}
    if cache_path.exists():
        try:
            result = json.loads(cache_path.read_text())
        except Exception:
            pass

    missing = [t for t in taxa if t.species not in result]
    if not missing:
        return result

    if client is None:
        api_key = os.environ.get(LLM_KEY_ENV)
        client = anthropic.Anthropic(api_key=api_key)

    for i in range(0, len(missing), batch):
        chunk = missing[i: i + batch]
        species_list = "\n".join(t.species for t in chunk)
        prompt = TRAIT_PROMPT.format(n=n, organ=organ, species_list=species_list)

        def _call() -> str:
            msg = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

        raw = _call()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Retry once
            raw = _call()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {}

        for species, traits in parsed.items():
            if isinstance(traits, list):
                result[species] = traits

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2))
    return result


def trait_prompts(species: str, traits: list[str]) -> list[str]:
    """[f"a photo of {species}, which has {t}" for t in traits] + [f"a photo of {species}"] (name prompt last)."""
    prompts = [f"a photo of {species}, which has {t}" for t in traits]
    prompts.append(f"a photo of {species}")
    return prompts


def build_text_protos(taxa: list[Taxon], traits: dict[str, list[str]],
                      encode_text: Callable[[list[str]], np.ndarray]) -> tuple[list[np.ndarray], list[list[str]]]:
    """For each taxon (in idx order): prompts = trait_prompts(...), protos[c] = L2-normalised encode_text(prompts).
    Species with no traits get only the name prompt. Returns (protos, prompt_strings) for scorers.TextScorer."""
    protos: list[np.ndarray] = []
    prompt_strings: list[list[str]] = []

    for taxon in taxa:
        species_traits = traits.get(taxon.species, [])
        prompts = trait_prompts(taxon.species, species_traits)
        vecs = encode_text(prompts)  # shape (len(prompts), D)
        # L2-normalise each row
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = (vecs / norms).astype(np.float32)
        protos.append(vecs)
        prompt_strings.append(prompts)

    return protos, prompt_strings
