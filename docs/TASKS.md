# Lanes and task cards

One lane = one folder (git worktree) on its own branch = one agent thread at a time. Lanes own disjoint files
(`docs/OWNERS`, enforced by a pre-commit hook), so merges are trivial. Paste a card's prompt into a **new thread**.
When a card is done, the agent writes a DONE entry in `worklog/<LANE>.md`, pushes its branch and stops; open a new
thread for the next card. Check progress any time with `python scripts/status.py --fetch`.

## Lane map

| Lane | Owns (only these files) | Needs first | T3 provider | Tests |
|---|---|---|---|---|
| **L1 data-eval** | `vanalens/data.py`, `eval/*`, `scripts/make_manifests.py`, `scripts/baseline_plantnet_api.py` | nothing | **Antigravity A** | `tests/test_eval.py` |
| **L2 encoders** (only lane allowed on the GPU) | `vanalens/encoders.py`, `scripts/baseline_bioclip.py`, `scripts/baseline_plantnet300k.py`, `scripts/extract_all.sh`, `notebooks/kaggle_extract.py` | L1-b manifests for real runs | **Antigravity A** (GPU) | `tests/test_encoders.py` (GPU) |
| **L3 core-ml** | `vanalens/scorers.py`, `fusion.py`, `conformal.py`, `hierarchy.py`, `scripts/fit_core.py` | nothing (synthetic tests); L2-c embeddings for L3-b | **Claude** | `tests/test_core.py` |
| **L4 context** | `vanalens/geo.py`, `traits.py`, `rerank.py`, `scripts/build_traits.py` | nothing; L2-a `encode_text` for L4-b protos | **Antigravity B** | `tests/test_context.py` |
| **L5 app** | `app/*`, `space/*` | nothing (uses `MockPipeline`) | **Antigravity B** | `tests/test_app.py` |
| **L6 integration** | `vanalens/pipeline.py` (except `MockPipeline`), `tests/test_pipeline.py`, `requirements.txt`, `README.md`, merges | L2-c, L3-b | **Claude** | `tests/test_pipeline.py` (L6 writes it) |
| Human | `AGENTS.md`, `CLAUDE.md`, `vanalens/contracts.py`, `vanalens/config.py`, this file | | | |

New tests go in new files you own; never edit someone else's test.

## Waves (maps onto the 24h plan)

| Hours | Run in parallel | Gate |
|---|---|---|
| 0-1 | Human: setup (`scripts/setup.ps1` on Windows), downloads started, DINOv3 license requested | `pytest --co` collects 38 tests |
| 1-6 | Claude: L3-a. Antigravity A: L1-a + L2-a. Antigravity B: L4-a + L5-a | each lane's tests green |
| 3-6 | L1-b, L2-b, L2-c (GPU background), L1-c | manifests + embeddings cached; baseline JSONs in `runs/` |
| 6-9 | Claude: L3-b then L6-a. Antigravity B: L4-b | **MVP: real pipeline answers a campus photo** |
| 9-16 | L6-b, L5-b, L2-d (optional) | final table `runs/L6/final.json` |
| 16-19 | fixes only | feature freeze at 19 |
| 19-24 | Pitch deck in Claude chat (Slides) from `runs/L6/final.json` | rehearse |

---

## L1 data-eval

**L1-a**
```
First write `L1` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L1-data-eval` and `git pull --rebase origin main`, then add a STARTED entry for L1-a to worklog/L1.md (format: AGENTS.md section 6).
You are lane L1 (data-eval) on VanaLens. Read AGENTS.md, then vanalens/contracts.py and vanalens/config.py only.
Implement eval/metrics.py, eval/harness.py, and these functions in vanalens/data.py: grouped_split,
cap_per_class, write_manifest, read_manifest, read_exif_gps. Follow the docstrings exactly.
Done when `pytest -q tests/test_eval.py` passes. Do not implement the dataset loaders yet.
When done: DONE entry in worklog/L1.md, commit with prefix [L1], `git push origin HEAD:lane/L1-data-eval`, stop.
```

**L1-b**
```
First write `L1` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L1-data-eval` and `git pull --rebase origin main`, then add a STARTED entry for L1-b to worklog/L1.md (format: AGENTS.md section 6).
Lane L1, card b. Read AGENTS.md. Implement load_dimpsar and load_campus in vanalens/data.py against the real
folders under $VANALENS_DATA (inspect with `find ... -maxdepth 2 | head -n 30`; never list everything).
Write scripts/make_manifests.py: for each dataset build grouped splits (gallery/cal/test), cap the gallery with
cap_per_class, write $VANALENS_CACHE/manifests/<dataset>_<split>.jsonl and $VANALENS_DATA/<dataset>/labels.json.
Print only per-split counts and number of classes.
When done: DONE entry in worklog/L1.md, commit with prefix [L1], `git push origin HEAD:lane/L1-data-eval`, stop.
```

**L1-c**
```
First write `L1` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L1-data-eval` and `git pull --rebase origin main`, then add a STARTED entry for L1-c to worklog/L1.md (format: AGENTS.md section 6).
Lane L1, card c. Read AGENTS.md. Write scripts/baseline_plantnet_api.py (baseline B0): sample 400 images (seed 0)
from a test manifest, call the Pl@ntNet API v2 identify endpoint (read my.plantnet.org docs; key from env
PLANTNET_API_KEY; pass the organ when known), cache each raw JSON under $VANALENS_CACHE/plantnet_api/, map returned
species names to taxa idx (unmatched -> uniform row), and save results with eval.harness to runs/L1/b0_<dataset>.json.
Never exceed 450 calls per day; skip images already cached.
When done: DONE entry in worklog/L1.md, commit with prefix [L1], `git push origin HEAD:lane/L1-data-eval`, stop.
```

## L2 encoders (GPU)

**L2-a**
```
First write `L2` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L2-encoders` and `git pull --rebase origin main`, then add a STARTED entry for L2-a to worklog/L2.md (format: AGENTS.md section 6).
You are lane L2 (encoders), the only lane allowed to use the GPU. Read AGENTS.md, vanalens/contracts.py,
vanalens/config.py. Implement vanalens/encoders.py per its docstring (BioCLIP 2 via open_clip + DINOv3 via
transformers, fp16, 6 GB GPU). If DINOv3 access is still pending, use the documented dinov2-large fallback.
Done when `VANALENS_GPU_TESTS=1 pytest -q tests/test_encoders.py` passes. Report images/second.
When done: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

**L2-b**
```
First write `L2` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L2-encoders` and `git pull --rebase origin main`, then add a STARTED entry for L2-b to worklog/L2.md (format: AGENTS.md section 6).
Lane L2, card b. Write scripts/baseline_bioclip.py (baseline B2): pybioclip CustomLabelsClassifier over the
dataset's species names on a test manifest -> (N, C) matrix in taxa order -> eval.harness ->
runs/L2/b2_<dataset>.json. Also record the open-ended TreeOfLifeClassifier top-1/top-5 species-name match rate.
When done: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

**L2-c**
```
First write `L2` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L2-encoders` and `git pull --rebase origin main`, then add a STARTED entry for L2-c to worklog/L2.md (format: AGENTS.md section 6).
Lane L2, card c. Write scripts/extract_all.sh that runs `python -m vanalens.encoders` for every manifest in
$VANALENS_CACHE/manifests/ into $VANALENS_CACHE/emb/<same name>.npz, in the background with nohup, logging to
runs/logs/. Then write notebooks/kaggle_extract.py: a standalone script for a Kaggle notebook (2x T4) that does the
same for Pl@ntNet-300K test + capped gallery. Start the local job; report only tail of the log.
When done: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

**L2-d (optional)**
```
First write `L2` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L2-encoders` and `git pull --rebase origin main`, then add a STARTED entry for L2-d to worklog/L2.md (format: AGENTS.md section 6).
Lane L2, card d. Write scripts/baseline_plantnet300k.py (baseline B1): load the pretrained ResNet weights from the
plantnet/PlantNet-300K repo, evaluate on the Pl@ntNet-300K test manifest, save runs/L2/b1_plantnet300k.json.
When done: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

## L3 core-ml

**L3-a**
```
First write `L3` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L3-core-ml` and `git pull --rebase origin main`, then add a STARTED entry for L3-a to worklog/L3.md (format: AGENTS.md section 6).
You are lane L3 (core-ml). Read AGENTS.md, vanalens/contracts.py, vanalens/config.py. Implement
vanalens/scorers.py, vanalens/fusion.py, vanalens/conformal.py and vanalens/hierarchy.py exactly as their
docstrings say (KNNScorer uses faiss.IndexFlatIP). No data, no GPU: use the synthetic fixtures.
Done when `pytest -q tests/test_core.py` passes.
When done: DONE entry in worklog/L3.md, commit with prefix [L3], `git push origin HEAD:lane/L3-core-ml`, stop.
```

**L3-b**
```
First write `L3` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L3-core-ml` and `git pull --rebase origin main`, then add a STARTED entry for L3-b to worklog/L3.md (format: AGENTS.md section 6).
Lane L3, card b. Write scripts/fit_core.py --dataset <name>: load gallery/cal/test npz from $VANALENS_CACHE/emb and
labels.json; fit ProbeScorer on gallery; compute kNN/probe(/text if protos exist) on cal; fit_weights + temperature
on cal; fit qhat (alpha from config) and an open-set threshold on cal; save
$VANALENS_CACHE/state/<dataset>.json; evaluate on test and write runs/L3/ablation_<dataset>.json with rows:
knn, probe, text, fused, fused+T, fused+T+sets. Print only the summary table.
When done: DONE entry in worklog/L3.md, commit with prefix [L3], `git push origin HEAD:lane/L3-core-ml`, stop.
```

## L4 context

**L4-a**
```
First write `L4` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L4-context` and `git pull --rebase origin main`, then add a STARTED entry for L4-a to worklog/L4.md (format: AGENTS.md section 6).
You are lane L4 (context). Read AGENTS.md, vanalens/contracts.py, vanalens/config.py. Implement the offline
functions: geo_prior and apply_prior (vanalens/geo.py); trait_prompts and build_text_protos (vanalens/traits.py);
should_rerank, build_rerank_prompt, parse_choice (vanalens/rerank.py). Done when `pytest -q tests/test_context.py`
passes. No network calls in this card.
When done: DONE entry in worklog/L4.md, commit with prefix [L4], `git push origin HEAD:lane/L4-context`, stop.
```

**L4-b**
```
First write `L4` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L4-context` and `git pull --rebase origin main`, then add a STARTED entry for L4-b to worklog/L4.md (format: AGENTS.md section 6).
Lane L4, card b. Implement the network parts with caching under $VANALENS_CACHE: gbif_species_counts and
resolve_gbif_keys (vanalens/geo.py; try lat 23.18, lon 79.99 and print only the top 10 counts);
generate_traits (anthropic SDK, key from VANALENS_LLM_KEY, model from env VANALENS_LLM_MODEL); and rerank() with
base64 images. Write scripts/build_traits.py to generate traits for a dataset's labels.json and save text protos
(using vanalens.encoders.Encoder.encode_text) to $VANALENS_CACHE/text/<dataset>.npz. Print 2 example species only.
When done: DONE entry in worklog/L4.md, commit with prefix [L4], `git push origin HEAD:lane/L4-context`, stop.
```

## L5 app

**L5-a**
```
First write `L5` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L5-app` and `git pull --rebase origin main`, then add a STARTED entry for L5-a to worklog/L5.md (format: AGENTS.md section 6).
You are lane L5 (app). Read AGENTS.md and vanalens/contracts.py. Implement app/app.py per its docstring with
Gradio, using only vanalens.pipeline.get_pipeline() and VANALENS_MOCK=1 during development. Mobile-friendly, no CDN.
Done when `pytest -q tests/test_app.py` passes and `VANALENS_MOCK=1 python -m app.app` serves the UI.
When done: DONE entry in worklog/L5.md, commit with prefix [L5], `git push origin HEAD:lane/L5-app`, stop.
```

**L5-b**
```
First write `L5` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L5-app` and `git pull --rebase origin main`, then add a STARTED entry for L5-b to worklog/L5.md (format: AGENTS.md section 6).
Lane L5, card b. Create space/ with what a Hugging Face Space needs to run app/app.py (entry file, requirements,
README front matter). Default to VANALENS_MOCK=0 and load state from the cache path given by env. Do not deploy;
print the exact commands for the human.
When done: DONE entry in worklog/L5.md, commit with prefix [L5], `git push origin HEAD:lane/L5-app`, stop.
```

## L6 integration

**L6-m (merge, repeat every ~2 h or when the board shows DONE)**: run in the MAIN checkout, not a worktree.
```
You are the L6 merge step, in the main checkout on branch main (there is no .lane file here). Read AGENTS.md.
1. `git fetch --all --quiet` then `python scripts/status.py`.
2. For each origin/lane/* branch with commits not in main: run
   `python scripts/check_owners.py --lane <LANE> --range main...origin/<branch>`; it must pass, otherwise skip that
   lane and record why. Then `git merge --no-ff origin/<branch> -m "merge <LANE>"`.
3. `pytest -q` (GPU tests skip). If a merge breaks tests, revert that merge commit and record it.
4. `git push origin main`.
5. Append one entry to worklog/MERGE.md (same format, card L6-m): merged lanes, skipped lanes + reason, test result.
Do not edit any lane's code. Stop.
```

**L6-a**
```
First write `L6` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L6-integration` and `git pull --rebase origin main`, then add a STARTED entry for L6-a to worklog/L6.md (format: AGENTS.md section 6).
You are lane L6 (integration). Read AGENTS.md, vanalens/contracts.py and the docstring of vanalens/pipeline.py.
Implement VanaLens (from_cache, predict, add_species) by calling the other lanes' public functions only.
Write tests/test_pipeline.py using the synthetic fixtures (monkeypatch the encoder). Then run end-to-end on
20 campus photos and save runs/L6/e2e_campus.json. Keep MockPipeline unchanged.
When done: DONE entry in worklog/L6.md, commit with prefix [L6], `git push origin HEAD:lane/L6-integration`, stop.
```

**L6-b**
```
First write `L6` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --rebase origin lane/L6-integration` and `git pull --rebase origin main`, then add a STARTED entry for L6-b to worklog/L6.md (format: AGENTS.md section 6).
Lane L6, card b. Merge all lane branches into main, run the full test suite, then build runs/L6/final.json:
every baseline (B0, B1, B2, B3 = probe only) and every ablation step on dimpsar, plantnet300k (subset) and campus,
with n per row. Print the table as markdown, nothing else.
When done: DONE entry in worklog/L6.md, commit with prefix [L6], `git push origin HEAD:lane/L6-integration`, stop.
```
