# VanaLens

Evidence-backed, uncertainty-aware plant species ID (approach F). Frozen BioCLIP 2 + DINOv3 encoders,
kNN / probe / text-trait scorers, fusion, geo prior, hierarchical backoff, conformal sets, gated VLM reranker.

This repo is a **multi-agent scaffold**: frozen contracts (`vanalens/contracts.py`), stubs with exact docstrings,
tests that run without data or a GPU (GPU tests are opt-in), and task cards per lane (`docs/TASKS.md`). Rules for
every agent are in `AGENTS.md` (Claude Code reads it through `CLAUDE.md`; Antigravity reads it directly).

## Setup

**Autopilot (agents finish it themselves): `docs/AUTOPILOT.md`.** Manual mode: `docs/SETUP_T3_ANTIGRAVITY.md`.

## Limits, honestly

- Several Claude Code threads draw from the **same** plan limit (5-hour window + weekly cap). Keep at most 2 Claude
  threads running; routine lanes run on Antigravity.
- CodeGraph saves the most on large existing codebases. This repo stays small, so the bigger savings come from
  short single-card threads, the no-data-dumps rule, synthetic tests and cached embeddings.

## Quick checks (PowerShell)

```powershell
& $env:VANALENS_PY scripts\status.py --fetch                 # who is doing what, open needs
& $env:VANALENS_PY scripts\next_cards.py --fetch             # what the dispatcher would start now
& $env:VANALENS_PY -m pytest -q tests\test_tooling.py        # autopilot tooling
$env:VANALENS_MOCK="1"; & $env:VANALENS_PY -m app.app         # UI with fake predictions
$env:VANALENS_GPU_TESTS="1"; & $env:VANALENS_PY -m pytest -q tests\test_encoders.py
```
