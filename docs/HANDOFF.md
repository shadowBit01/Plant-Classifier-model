# HANDOFF: context for the lead Claude thread in T3 Code

Written 15 Nov 2026, 02:05 IST, when work moved from a claude.ai chat into T3 Code. The lead thread can't see that
chat; this file is the record of what was decided. Human-owned: only the human or the dispatcher (on `main`)
edits it, and every change gets a line in the decisions log at the end.

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
6. Baselines: B0 Pl@ntNet API (production; fair on the inat_india test split, DIMPSAR and campus), B1 Pl@ntNet-300K
   repo ResNet weights (only if L2-d is enabled), B2 BioCLIP 2 zero-shot, B3 DINOv3 linear probe, B4 general VLM open-ended (expect about 17%).
7. Metrics: top-1, top-5, macro top-1, set size at 90% coverage, ECE, genus/family accuracy, open-set AUROC,
   few-shot curve (1/5/10), latency.
8. Caveat to say on the results slide: BioCLIP 2 trained on GBIF data, and Pl@ntNet publishes to GBIF, so
   older or Pl@ntNet-derived images may be in its training data. Headline numbers come from the post-July-2025 test split.

## Setup facts
- One machine: Lenovo LOQ, RTX 3050 6 GB, 16 GB RAM, **Windows** (PowerShell 5.1; Claude Code uses Git Bash).
  Setup via `scripts/setup.ps1`; venv Python is `$VANALENS_PY`.
- Repo: GitHub `Plant-Classifier-model`. Scaffold = frozen `vanalens/contracts.py`, stubs with exact docstrings,
  tests, `docs/TASKS.md` task cards, `docs/PLAN.json` (cards, deps, lanes, providers), `docs/OWNERS` + pre-commit
  hook, `worklog/<LANE>.md`, `scripts/status.py` board.
- Autopilot: a Claude dispatcher (`docs/DISPATCHER.md`) runs every 30 min in the main checkout (the single T3
  project `Plant-Classifier-model`, Local mode). `scripts/next_cards.py` decides what can start; `scripts/merge_lanes.py` merges finished lanes
  behind ownership + gate tests; `scripts/card_prompt.py` prints the exact prompt; threads start through T3's MCP.
- One T3 project only. Each lane works in `.lanes/LX` inside it (git worktree made by `scripts/lane.py`, local
  branch `lanework/LX`, pushed to `lane/LX-...`). Lanes merge main in, never rebase. The root stays on main and its
  pre-commit hook accepts only human-owned files there.
- T3 providers: Claude (dispatcher, L3 core-ml, L6 integration, L7 pitch), Antigravity A (L1 data-eval, L2 encoders,
  only lane on the GPU), Antigravity B (L4 context, L5 app). At most 2 threads per provider. Claude threads share
  one plan limit. Lanes are tied to files, not accounts: the dispatcher can move a lane to another provider.
- CodeGraph installed; agents run `codegraph init` in their folder if missing.
- `VANALENS_LLM_KEY` (not `ANTHROPIC_API_KEY`) is for runtime LLM calls; it may be empty. If empty: L4 writes the
  traits JSON itself, and the reranker and B4 are cut.

## Background docs
- `docs/RESEARCH.md`: full research report (datasets with links, 15 papers, baselines, system design, 24h
  schedule, tooling, 8-slide pitch outline, risks). Read only the section you need; do not load it whole.
- `docs/AUTOPILOT.md`: one-time setup for autopilot. `docs/SETUP_T3_ANTIGRAVITY.md`: manual mode.

## Plan and gates
| Hours | Work | Gate |
|---|---|---|
| 0-1 | Setup, one T3 project, schedule | tooling tests green, smoke test dispatches 5 cards |
| 1-6 | Wave 1: L1-a, L2-a, L3-a, L4-a, L5-a; then L1-b (inat_india), L2-b, L2-c, L1-c | lane tests green, embeddings, baseline JSONs |
| 6-9 | L3-b, L4-b, L6-a | **MVP: real pipeline answers an inat_india test photo** |
| 9-16 | L6-b final table, L5-b Space | `runs/L6/final.json` |
| 16-19 | Fixes only | **feature freeze at hour 19** |
| 19-24 | L7-a deck (`pitch/VanaLens.pptx`, script, demo steps); human rehearses | submit |

Cut ladder if behind: reranker, then geo prior, then DINOv3 (BioCLIP 2 only), then text prototypes.
Never cut: baseline table, conformal sets, evidence panel, inat_india test split.

## Open human to-dos
- Follow `docs/AUTOPILOT.md` once. Optional: `setx PLANTNET_API_KEY`, `setx VANALENS_LLM_KEY`, DINOv3 license,
  DIMPSAR or campus photos in `$VANALENS_DATA` before L1-b runs.
- Act on any `Needs: HUMAN:` line in `worklog/DISPATCH.md`. Rehearse with the deck at the end.

## How the human likes to work
- Direct, terse, actionable specifics. No em dashes in any writing.
- Wants every agent to keep a worklog and never overlap with another lane.

## Decisions log (append below, newest last)
- 02:05 Work moved to T3 Code. Lead thread = Claude in the main checkout (no worktree).
- 02:10 Machine is Windows: added setup.ps1, .gitattributes (LF for hooks), Python-finding hook, UTF-16-safe .lane.
- 02:50 Autopilot mode: dispatcher (docs/DISPATCHER.md) runs on a schedule, starts cards from docs/PLAN.json via
  T3 MCP, merges, routes BLOCKED items, may edit human-owned files with a logged decision. Added L7 pitch lane.
  inat_india replaces manual downloads as the primary dataset.
- 03:40 Final audit: merges go through scripts/merge_lanes.py (lane gate tests, abort on red, failures routed
  as fix cards); lanes merge main instead of rebasing; B4 added to L4-b; L6-b no longer merges.
- 04:10 Single T3 project: lanes work in .lanes/LX worktrees created by scripts/lane.py (which also logs STARTED);
  PCM-L* projects and make_worktrees.ps1 retired; commits on main limited to human-owned files.
- 2026-11-15 14:40 Wave 2 (human request, via supervisor): new dataset vl600 (~400 top India + ~200 top world
  plant species + a fixed common-Indian list, selection by observation count only, up to 80 pre-2025-07-01 photos
  + 15 post-cutoff test obs per species, iNat open-data bucket). New cards L1-d, L2-e, L3-c, L4-c, L5-c, L6-c,
  L7-b, L7-c. L7-a deps -> L6-c (already DONE, so L7-b also depends on L6-c and rebuilds the deck). L3-c also
  waits for L4-c so fusion includes text. L3-c may use the GPU (L2 has no GPU work after L2-e). Report lives in
  pitch/report/ (owned by L7) instead of report/, which has no owner.
- 2026-11-15 14:45 Human: disk is not a constraint (L1-d cap lifted, stop only under 10 GB free); campus
  likely not coming, vl600 is the main dataset; accuracy target 95%+. Added honest accuracy levers: L2-e flip TTA
  + 2 augmented gallery views (X_aug/y_aug, extra npz keys), L3-c aug training, label smoothing, ensemble,
  all chosen on cal; L6-c reports which metrics reach 95%. No species dropped, no tuning on test.
- 2026-11-15 17:58 stale_minutes 120 -> 230 (240 breaks test_next_cards_stale_and_blocked): L2-e (61k images, flip TTA, ~12 img/s) is healthy at 1h45m and
  would be re-dispatched at 18:12, putting a second GPU job on the 6 GB card. Long vl600 jobs need more headroom.
- 2026-11-15 18:30 Human: the deck was weak and a project report is required for submission. L7 split into
  L7-b (deck redesign: clean white/green theme, 11-13 slides, slides exported to PNG via PowerPoint COM and checked),
  L7-c (project report md -> PDF via headless Edge or Word: how it works, methods, dataset, results, limitations,
  plan to overcome them), L7-d (skeptical review of both + demo.md). Report builder lives in pitch/report/ (L7-owned).
- 2026-11-15 19:55 lead (human approved): DINO weight-loading bug. DINOv3 is gated (no HF login) so the encoder uses
  dinov2-large, but the custom safetensors loader skips 192/439 tensors (all attention weights; the new transformers
  renamed them), so the DINO half of every cached embedding is random and differs from the live encoder (live vs cache
  DINO cos 0.3-0.58, BioCLIP 1.00; live open-set rejects ~95%). Added L2-f (fix loader + test + recompute DINO block for
  all caches), L3-d (refit on cal), L6-d (re-run final.json + live e2e); L7-c now also depends on L6-d and refreshes
  the deck numbers. Reported results so far (vl600 1-photo .939) are effectively BioCLIP-only.
- 2026-11-15 23:58 dispatcher: L2-f thread failed with "Antigravity quota reached" (fix_dino job died at 20:58, vl600_gallery shard 9/20); L2 provider -> Claude in PLAN.json, L2-f re-dispatched with --resume.
- 2026-11-16 00:05 lead: L2-f lost ~3 h (Antigravity quota at 20:58, caught as stale at 23:59). To protect the 05:45
  deadline: L7-c (report) now depends only on L7-b and runs in parallel with L2-f/L3-d/L6-d, writing numbers through a
  generated include; L7-d now depends on L7-c + L6-d and first refreshes all deck/report numbers. stale_minutes 230 -> 120
  so a dead thread is caught within 2 h (remaining cards are all < 2 h).
