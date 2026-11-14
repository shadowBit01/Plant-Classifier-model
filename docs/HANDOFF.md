# HANDOFF: context for the lead Claude thread in T3 Code

Written 15 Nov 2026, 02:05 IST, when work moved from a claude.ai chat into T3 Code. The lead thread can't see that
chat; this file is the record of what was decided. Human-owned: the lead edits it on `main` only when asked.

## The project
- 24-hour build, research to training to final pitch: a plant species classifier. Input: one or more JPEG/PNG
  photos (leaf, flower, whole plant, bark, fruit). Output: species, with honest confidence.
- Brief from the human: fine-tuning alone would be underwhelming; pivot toward ideas that are somewhat novel.
- Working name: VanaLens (placeholder).

## Decisions (do not reopen without the human)
1. **Approach F**, no fine-tuning as the core. Frozen encoders: BioCLIP 2 (`hf-hub:imageomics/bioclip-2`, ViT-L/14,
   768-d) + DINOv3 ViT-L/16 (1024-d, gated on Hugging Face; fallback `facebook/dinov2-large`, also 1024-d).
2. Scorers: kNN over a reference gallery (gives evidence photos), logistic-regression probe, text prototypes from
   LLM-written visual traits ("classification by description"). Log-linear fusion + temperature scaling.
3. Then: GBIF geo prior from photo GPS, multi-photo fusion, hierarchical backoff (species, genus, family),
   split-conformal 90% prediction sets, open-set "not in our flora" reject, VLM reranker only when the set has
   more than one species (RAR-style, choose from a short list; VLMs are weak open-ended).
4. Pitch angle: add a species with 5 photos and no retraining; honest uncertainty with a guarantee; evidence
   photos + matched traits; location-aware; a second photo shrinks the set.
5. Eval sets: **inat_india** (built automatically by L1-b from the iNaturalist API: 60 common Madhya Pradesh plant
   species, GPS on every photo, test = observations from July 2025 on, after BioCLIP 2's data cutoff, so
   contamination-free; also the set where the geo prior is measured). DIMPSAR and campus photos are optional
   extras if they appear in `$VANALENS_DATA`. Pl@ntNet-300K (L2-d) is disabled by default.
6. Baselines: B0 Pl@ntNet API (production; unfair on Pl@ntNet-300K, fair on DIMPSAR + campus), B1 Pl@ntNet-300K
   repo ResNet weights, B2 BioCLIP 2 zero-shot, B3 DINOv3 linear probe, B4 general VLM open-ended (expect about 17%).
7. Metrics: top-1, top-5, macro top-1, set size at 90% coverage, ECE, genus/family accuracy, open-set AUROC,
   few-shot curve (1/5/10), latency.
8. Caveat to say on the results slide: BioCLIP 2 trained on GBIF data, and Pl@ntNet publishes to GBIF, so
   older or Pl@ntNet-derived images may be in its training data. Headline numbers come from the post-July-2025 test split.

## Setup facts
- One machine: Lenovo LOQ, RTX 3050 6 GB, 16 GB RAM, **Windows** (PowerShell 5.1; Claude Code uses Git Bash).
  Setup via `scripts/setup.ps1`; venv Python is `$VANALENS_PY`. Big embedding jobs can go to Kaggle (2x T4).
- Repo: GitHub `Plant-Classifier-model`. Scaffold = frozen `vanalens/contracts.py`, stubs with exact
  docstrings, 38 tests, `docs/TASKS.md` task cards, `docs/OWNERS` + pre-commit hook, `worklog/<LANE>.md`,
  `scripts/status.py`, merge card L6-m.
- T3 providers: Claude (L3 core-ml, L6 integration, L6-m merges, lead), Antigravity A (L1 data-eval, L2 encoders,
  only lane on the GPU), Antigravity B (L4 context, L5 app). At most 2 threads per provider. Claude threads share
  one plan limit. Lanes are tied to files, not accounts: if one Antigravity account runs out, the other continues.
- CodeGraph installed; agents run `codegraph init` in their worktree if missing.
- `VANALENS_LLM_KEY` (not `ANTHROPIC_API_KEY`) is for runtime LLM calls; it may be empty. If empty: a Claude thread
  writes the traits JSON directly, and the reranker is cut.

## Background docs
- `docs/RESEARCH.md`: full research report (datasets with links, 15 papers, baselines, system design, 24h
  schedule, tooling, 8-slide pitch outline, risks). Read only the section you need; do not load it whole.
- `docs/SETUP_T3_ANTIGRAVITY.md`: machine and provider setup.

## Plan and gates
| Hours | Work | Gate |
|---|---|---|
| 0-1 | Setup, downloads, DINOv3 license | tests collect, branches pushed |
| 1-6 | Wave 1: L3-a, L1-a, L2-a, L4-a, L5-a; then L1-b, L2-b, L2-c, L1-c | lane tests green, baseline JSONs |
| 6-9 | L3-b, L6-a, L4-b | **MVP: real pipeline answers a campus photo** |
| 9-16 | L6-b final table, L5-b Space, optional L2-d | `runs/L6/final.json` |
| 16-19 | Fixes only | **feature freeze at hour 19** |
| 19-24 | Pitch deck (outline in RESEARCH.md section 10), demo video, rehearsal | submit |

Cut ladder if behind: reranker, then geo prior, then DINOv3 (BioCLIP 2 only), then text prototypes.
Never cut: baseline table, conformal sets, evidence panel, campus test.

## Open human to-dos (as of handoff)
- Copy scaffold into the repo, run `scripts/setup.ps1`, push all branches, set API-key vars with setx, relaunch T3.
- Accept the DINOv3 license, `hf auth login`.
- Download DIMPSAR to `~/vanalens-data/dimpsar/` (unlocks L1-b). Optional: Pl@ntNet-300K (31.7 GB).
- Pl@ntNet API key. Campus photos in daylight: `~/vanalens-data/campus/<Genus species>/*.jpg`, GPS on.

## How the human likes to work
- Direct, terse, actionable specifics. No em dashes in any writing.
- Wants every agent to keep a worklog and never overlap with another lane.

## Decisions log (append below, newest last)
- 02:05 Work moved to T3 Code. Lead thread = Claude in the main checkout (no worktree).
- 02:10 Machine is Windows: added setup.ps1, .gitattributes (LF for hooks), Python-finding hook, UTF-16-safe .lane.
- 02:50 Autopilot mode: dispatcher (docs/DISPATCHER.md) runs on a schedule, starts cards from docs/PLAN.json via
  T3 MCP, merges, routes BLOCKED items, may edit human-owned files with a logged decision. Added L7 pitch lane.
  inat_india replaces manual downloads as the primary dataset.
