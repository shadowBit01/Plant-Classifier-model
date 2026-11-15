"""Lane L4 (context). Uncertainty-gated VLM reranker (RAR-style). Tests: tests/test_context.py

The VLM only chooses among candidates; it never names a species freely (VLMs are weak open-ended).
Any error, timeout or unparseable answer -> fall back to the first candidate, reranked=False.
"""
from __future__ import annotations

import re

from .contracts import Taxon


def should_rerank(pred_set: list[int], max_candidates: int = 5) -> bool:
    """True when the conformal set holds more than one species. (Reranking uses the top `max_candidates`.)"""
    return len(pred_set) > 1


def build_rerank_prompt(candidates: list[Taxon], traits: dict[str, list[str]]) -> str:
    """Text part of the request: numbered candidates with their traits, then the instruction:
    pick exactly one listed species, cite 2 traits visible in the query photo,
    end with a final line 'ANSWER: <species name>'."""
    lines: list[str] = []
    lines.append("You are a plant identification expert. Look at the query photo and choose the best match.")
    lines.append("")
    lines.append("Candidates:")
    for i, taxon in enumerate(candidates, start=1):
        species_traits = traits.get(taxon.species, [])
        trait_str = "; ".join(species_traits) if species_traits else "no traits available"
        lines.append(f"  {i}. {taxon.species} — {trait_str}")
    lines.append("")
    lines.append("Instructions:")
    lines.append("  - Pick exactly one of the listed species.")
    lines.append("  - Cite 2 traits visible in the query photo that support your choice.")
    lines.append("  - End your response with a final line in exactly this format:")
    lines.append("ANSWER: <species name>")
    return "\n".join(lines)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_choice(text: str, candidates: list[Taxon]) -> int | None:
    """Find the last 'ANSWER:' line; match case-insensitively (ignore extra spaces/punctuation) to a
    candidate species. Return that Taxon.idx, or None."""
    answer_text: str | None = None
    for line in text.splitlines():
        m = re.match(r"(?i)ANSWER\s*:\s*(.*)", line.strip())
        if m:
            answer_text = m.group(1).strip()
    # Use last ANSWER found
    if answer_text is None:
        return None

    normalized_answer = _normalize(answer_text)
    for taxon in candidates:
        if _normalize(taxon.species) == normalized_answer:
            return taxon.idx
    return None


def rerank(query_paths: list[str], candidates: list[Taxon], ref_paths: dict[int, list[str]],
           traits: dict[str, list[str]], client=None) -> tuple[int, str, bool]:
    """Send query photo(s) + up to 2 reference photos per candidate (ref_paths[taxon.idx]) + prompt.
    Return (chosen idx, rationale text, reranked flag). Key from VANALENS_LLM_KEY."""
    import base64
    import os

    import anthropic

    from .config import LLM_KEY_ENV

    if not candidates:
        return -1, "no candidates", False

    fallback_idx = candidates[0].idx

    if client is None:
        api_key = os.environ.get(LLM_KEY_ENV)
        if not api_key:
            return fallback_idx, "reranker disabled: no key", False
        client = anthropic.Anthropic(api_key=api_key)

    model = os.environ.get("VANALENS_LLM_MODEL") or "claude-3-5-sonnet-20241022"

    def _img_block(path: str) -> dict | None:
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            ext = path.rsplit(".", 1)[-1].lower() if "." in path else "jpeg"
            media_types = {
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png",
                "webp": "image/webp",
                "gif": "image/gif",
            }
            media_type = media_types.get(ext, "image/jpeg")
            return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
        except Exception:
            return None

    prompt_text = build_rerank_prompt(candidates, traits)

    content: list = []
    for qp in query_paths:
        blk = _img_block(qp)
        if blk:
            content.append(blk)

    for taxon in candidates:
        refs = ref_paths.get(taxon.idx, [])[:2]
        for rp in refs:
            blk = _img_block(rp)
            if blk:
                content.append(blk)

    content.append({"type": "text", "text": prompt_text})

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": content}],
        )
        rationale = msg.content[0].text
        chosen = parse_choice(rationale, candidates)
        if chosen is None:
            return fallback_idx, rationale, False
        return chosen, rationale, True
    except Exception as e:
        return fallback_idx, str(e), False

