# VanaLens

Evidence-backed, uncertainty-aware plant species ID (approach F). Frozen BioCLIP 2 + DINOv3 encoders,
kNN / probe / text-trait scorers, fusion, geo prior, hierarchical backoff, conformal sets, gated VLM reranker.

This repo is a **multi-agent scaffold**: frozen contracts (`vanalens/contracts.py`), stubs with exact docstrings,
38 tests (35 run without data or a GPU), and task cards per lane (`docs/TASKS.md`). Rules for every agent are in
`AGENTS.md` (Claude Code reads it through `CLAUDE.md`; Antigravity reads it directly).

## Setup

Follow **`docs/SETUP_T3_ANTIGRAVITY.md`**: Claude + Antigravity A + Antigravity B as T3 Code providers on one machine,
one worktree + branch per lane, ownership hook, worklogs and a status board.

## Limits, honestly

- Several Claude Code threads draw from the **same** plan limit (5-hour window + weekly cap). Keep at most 2 Claude
  threads running; routine lanes run on Antigravity.
- CodeGraph saves the most on large existing codebases. This repo stays small, so the bigger savings come from
  short single-card threads, the no-data-dumps rule, synthetic tests and cached embeddings.

## Quick checks

```bash
python scripts/status.py --fetch           # who is doing what, open needs
pytest -q tests/test_core.py              # L3
VANALENS_MOCK=1 python -m app.app         # UI with fake predictions
VANALENS_GPU_TESTS=1 pytest -q tests/test_encoders.py
```
