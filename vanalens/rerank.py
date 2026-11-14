"""Lane L4 (context). Uncertainty-gated VLM reranker (RAR-style). Tests: tests/test_context.py

The VLM only chooses among candidates; it never names a species freely (VLMs are weak open-ended).
Any error, timeout or unparseable answer -> fall back to the first candidate, reranked=False.
"""
from __future__ import annotations

from .contracts import Taxon


def should_rerank(pred_set: list[int], max_candidates: int = 5) -> bool:
    """True when the conformal set holds more than one species. (Reranking uses the top `max_candidates`.)"""
    raise NotImplementedError


def build_rerank_prompt(candidates: list[Taxon], traits: dict[str, list[str]]) -> str:
    """Text part of the request: numbered candidates with their traits, then the instruction:
    pick exactly one listed species, cite 2 traits visible in the query photo,
    end with a final line 'ANSWER: <species name>'."""
    raise NotImplementedError


def parse_choice(text: str, candidates: list[Taxon]) -> int | None:
    """Find the last 'ANSWER:' line; match case-insensitively (ignore extra spaces/punctuation) to a
    candidate species. Return that Taxon.idx, or None."""
    raise NotImplementedError


def rerank(query_paths: list[str], candidates: list[Taxon], ref_paths: dict[int, list[str]],
           traits: dict[str, list[str]], client=None) -> tuple[int, str, bool]:
    """Send query photo(s) + up to 2 reference photos per candidate (ref_paths[taxon.idx]) + prompt.
    Return (chosen idx, rationale text, reranked flag). Key from VANALENS_LLM_KEY."""
    raise NotImplementedError
