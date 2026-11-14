# Lanes and task cards

One lane = one fixed folder (git worktree `PCM-LX`, also a T3 project) on its own branch = one agent thread at a time.
In autopilot mode the dispatcher (docs/DISPATCHER.md) starts these cards itself, using docs/PLAN.json. Lanes own disjoint files
(`docs/OWNERS`, enforced by a pre-commit hook), so merges are trivial. Paste a card's prompt into a **new thread**.
When a card is done, the agent writes a DONE entry in `worklog/<LANE>.md`, pushes its branch and stops; the dispatcher
merges it (`scripts/merge_lanes.py`) and opens a new thread for the next card. Check progress any time with `python scripts/status.py --fetch`.

## Lane map

| Lane | Owns (only these files) | Needs first | T3 provider | Tests |
|---|---|---|---|---|
| **L1 data-eval** | `vanalens/data.py`, `eval/*`, `scripts/make_manifests.py`, `scripts/baseline_plantnet_api.py` | nothing | **Antigravity A** | `tests/test_eval.py` |
| **L2 encoders** (only lane allowed on the GPU) | `vanalens/encoders.py`, `scripts/baseline_bioclip.py`, `scripts/baseline_plantnet300k.py`, `scripts/extract_all.py`, `notebooks/kaggle_extract.py` | L1-b manifests for real runs | **Antigravity A** (GPU) | `tests/test_encoders.py` (GPU) |
| **L3 core-ml** | `vanalens/scorers.py`, `fusion.py`, `conformal.py`, `hierarchy.py`, `scripts/fit_core.py` | nothing (synthetic tests); L2-c embeddings for L3-b | **Claude** | `tests/test_core.py` |
| **L4 context** | `vanalens/geo.py`, `traits.py`, `rerank.py`, `scripts/build_traits.py`, `scripts/baseline_vlm.py` | nothing; L2-a `encode_text` for L4-b protos | **Antigravity B** | `tests/test_context.py` |
| **L5 app** | `app/*`, `space/*` | nothing (uses `MockPipeline`) | **Antigravity B** | `tests/test_app.py` |
| **L6 integration** | `vanalens/pipeline.py` (except `MockPipeline`), `tests/test_pipeline.py`, `requirements.txt`, `README.md` | L2-c, L3-b | **Claude** | `tests/test_pipeline.py` (L6 writes it) |
| **L7 pitch** | `pitch/*`, `scripts/build_deck.py` | L6-b final table | **Claude** | none |
| Human / dispatcher | `AGENTS.md`, `CLAUDE.md`, `docs/*`, `vanalens/contracts.py`, `vanalens/config.py`, tooling scripts; merges via `scripts/merge_lanes.py` | | Claude (PCM-main) | `tests/test_tooling.py` |

New tests go in new files you own; never edit someone else's test.

## Waves (maps onto the 24h plan)

| Hours | Run in parallel | Gate |
|---|---|---|
| 0-1 | Human: `docs/AUTOPILOT.md` (setup, worktrees, T3 projects, schedule) | `pytest --co` collects 48 tests; `tests/test_tooling.py` 14 passed |
| 1-6 | Claude: L3-a. Antigravity A: L1-a + L2-a. Antigravity B: L4-a + L5-a | each lane's tests green |
| 3-6 | L1-b, then L2-b, L2-c (GPU), L1-c | inat_india manifests + embeddings cached; baseline JSONs in `runs/` |
| 6-9 | Claude: L3-b then L6-a. Antigravity B: L4-b | **MVP: real pipeline answers an inat_india test photo** |
| 9-16 | L6-b, L5-b | final table `runs/L6/final.json` |
| 16-19 | fixes only | feature freeze at 19 |
| 19-24 | Claude: L7-a builds `pitch/VanaLens.pptx` from `runs/L6/final.json` | human rehearses |

---

## L1 data-eval

**L1-a**
```
First write `L1` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L1-data-eval` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L1-a to worklog/L1.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L1] start L1-a"`, `git push origin HEAD:lane/L1-data-eval`) so the dispatcher sees the card is in progress.
You are lane L1 (data-eval) on VanaLens. Read AGENTS.md, then vanalens/contracts.py and vanalens/config.py only.
Implement eval/metrics.py, eval/harness.py, and these functions in vanalens/data.py: grouped_split,
cap_per_class, write_manifest, read_manifest, read_exif_gps. Follow the docstrings exactly.
Done when `pytest -q tests/test_eval.py` passes. Do not implement the dataset loaders yet.
When done: DONE entry in worklog/L1.md, commit with prefix [L1], `git push origin HEAD:lane/L1-data-eval`, stop.
```

**L1-b**
```
First write `L1` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L1-data-eval` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L1-b to worklog/L1.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L1] start L1-b"`, `git push origin HEAD:lane/L1-data-eval`) so the dispatcher sees the card is in progress.
Lane L1, card b. Read AGENTS.md. Build the evaluation datasets (all automatic; nobody can download for you).
1. inat_india (always): with the iNaturalist API v1 (api.inaturalist.org; max 1 request/s; set a User-Agent),
   resolve the place id for Madhya Pradesh, India, then take the 60 most-observed research-grade species under
   taxon_id 47126 (Plantae) via observations/species_counts. Per species fetch research-grade observations with
   coordinates and a CC0 / CC-BY / CC-BY-NC photo: up to 30 observed before 2025-06-01 (gallery + cal) and up to 15
   observed on or after 2025-07-01 (test: after BioCLIP 2's training data was collected, so contamination-free).
   One medium photo per observation. Drop species with fewer than 5 test observations. Save
   $VANALENS_DATA/inat_india/<Genus species>/<obs_id>.jpg and meta.csv (image,lat,lon,obs_id,observed_on,license,url).
   Add load_inat_india to vanalens/data.py (source "inat", group = obs id, lat/lon filled).
2. dimpsar: only if $VANALENS_DATA/dimpsar exists, implement load_dimpsar. campus: only if $VANALENS_DATA/campus
   exists, implement load_campus. Otherwise skip them and say so in the DONE entry.
3. scripts/make_manifests.py: for each available dataset write $VANALENS_CACHE/manifests/<dataset>_<split>.jsonl
   (gallery/cal/test; inat_india uses the date rule, cal = 20% of pre-cutoff observations by group) and
   $VANALENS_DATA/<dataset>/labels.json; cap gallery with cap_per_class. Print only per-split counts and classes.
When done: DONE entry in worklog/L1.md, commit with prefix [L1], `git push origin HEAD:lane/L1-data-eval`, stop.
```

**L1-c**
```
First write `L1` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L1-data-eval` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L1-c to worklog/L1.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L1] start L1-c"`, `git push origin HEAD:lane/L1-data-eval`) so the dispatcher sees the card is in progress.
Lane L1, card c. Read AGENTS.md. Write scripts/baseline_plantnet_api.py (baseline B0): sample 400 images (seed 0)
from a test manifest, call the Pl@ntNet API v2 identify endpoint (read my.plantnet.org docs; key from env
PLANTNET_API_KEY; pass the organ when known), cache each raw JSON under $VANALENS_CACHE/plantnet_api/, map returned
species names to taxa idx (unmatched -> uniform row), and save results with eval.harness to runs/L1/b0_<dataset>.json.
Never exceed 450 calls per day; skip images already cached. If PLANTNET_API_KEY is empty, write the script,
skip the run, and mark the DONE entry 'B0 skipped: no key'.
When done: DONE entry in worklog/L1.md, commit with prefix [L1], `git push origin HEAD:lane/L1-data-eval`, stop.
```

## L2 encoders (GPU)

**L2-a**
```
First write `L2` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L2-encoders` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L2-a to worklog/L2.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L2] start L2-a"`, `git push origin HEAD:lane/L2-encoders`) so the dispatcher sees the card is in progress.
You are lane L2 (encoders), the only lane allowed to use the GPU. Read AGENTS.md, vanalens/contracts.py,
vanalens/config.py. Implement vanalens/encoders.py per its docstring (BioCLIP 2 via open_clip + DINOv3 via
transformers, fp16, 6 GB GPU). If DINOv3 access is still pending, use the documented dinov2-large fallback.
Done when `VANALENS_GPU_TESTS=1 pytest -q tests/test_encoders.py` passes (PowerShell:
`$env:VANALENS_GPU_TESTS="1"; & $env:VANALENS_PY -m pytest -q tests/test_encoders.py`). Report images/second.
When done: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

**L2-b**
```
First write `L2` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L2-encoders` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L2-b to worklog/L2.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L2] start L2-b"`, `git push origin HEAD:lane/L2-encoders`) so the dispatcher sees the card is in progress.
Lane L2, card b. Write scripts/baseline_bioclip.py (baseline B2): pybioclip CustomLabelsClassifier over the
dataset's species names on a test manifest -> (N, C) matrix in taxa order -> eval.harness ->
runs/L2/b2_<dataset>.json. Also record the open-ended TreeOfLifeClassifier top-1/top-5 species-name match rate.
When done: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

**L2-c**
```
First write `L2` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L2-encoders` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L2-c to worklog/L2.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L2] start L2-c"`, `git push origin HEAD:lane/L2-encoders`) so the dispatcher sees the card is in progress.
Lane L2, card c. Write scripts/extract_all.py (cross-platform, stdlib + vanalens): for every manifest in
$VANALENS_CACHE/manifests/ without an up-to-date $VANALENS_CACHE/emb/<same name>.npz, run
`sys.executable -m vanalens.encoders --manifest ... --out ...` one after another, logging to
runs/logs/extract_all.log and printing "ALL DONE" at the end. `--detach` relaunches itself in the background
(Windows: subprocess creationflags DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP; POSIX: start_new_session=True).
Start it with --detach, then check the last 5 log lines every few minutes until "ALL DONE". Write the DONE entry
only when every manifest has its .npz (check with a one-line python -c that prints names and shapes).
When done: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

**L2-d (optional)**
```
First write `L2` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L2-encoders` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L2-d to worklog/L2.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L2] start L2-d"`, `git push origin HEAD:lane/L2-encoders`) so the dispatcher sees the card is in progress.
Lane L2, card d. Write scripts/baseline_plantnet300k.py (baseline B1): load the pretrained ResNet weights from the
plantnet/PlantNet-300K repo, evaluate on the Pl@ntNet-300K test manifest, save runs/L2/b1_plantnet300k.json.
When done: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

## L3 core-ml

**L3-a**
```
First write `L3` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L3-core-ml` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L3-a to worklog/L3.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L3] start L3-a"`, `git push origin HEAD:lane/L3-core-ml`) so the dispatcher sees the card is in progress.
You are lane L3 (core-ml). Read AGENTS.md, vanalens/contracts.py, vanalens/config.py. Implement
vanalens/scorers.py, vanalens/fusion.py, vanalens/conformal.py and vanalens/hierarchy.py exactly as their
docstrings say (KNNScorer uses faiss.IndexFlatIP). No data, no GPU: use the synthetic fixtures.
Done when `pytest -q tests/test_core.py` passes.
When done: DONE entry in worklog/L3.md, commit with prefix [L3], `git push origin HEAD:lane/L3-core-ml`, stop.
```

**L3-b**
```
First write `L3` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L3-core-ml` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L3-b to worklog/L3.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L3] start L3-b"`, `git push origin HEAD:lane/L3-core-ml`) so the dispatcher sees the card is in progress.
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
First write `L4` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L4-context` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L4-a to worklog/L4.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L4] start L4-a"`, `git push origin HEAD:lane/L4-context`) so the dispatcher sees the card is in progress.
You are lane L4 (context). Read AGENTS.md, vanalens/contracts.py, vanalens/config.py. Implement the offline
functions: geo_prior and apply_prior (vanalens/geo.py); trait_prompts and build_text_protos (vanalens/traits.py);
should_rerank, build_rerank_prompt, parse_choice (vanalens/rerank.py). Done when `pytest -q tests/test_context.py`
passes. No network calls in this card.
When done: DONE entry in worklog/L4.md, commit with prefix [L4], `git push origin HEAD:lane/L4-context`, stop.
```

**L4-b**
```
First write `L4` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L4-context` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L4-b to worklog/L4.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L4] start L4-b"`, `git push origin HEAD:lane/L4-context`) so the dispatcher sees the card is in progress.
Lane L4, card b. Implement the network parts with caching under $VANALENS_CACHE: gbif_species_counts and
resolve_gbif_keys (vanalens/geo.py; try lat 23.18, lon 79.99 and print only the top 10 counts);
generate_traits (anthropic SDK, key from VANALENS_LLM_KEY, model from env VANALENS_LLM_MODEL); and rerank() with
base64 images. Write scripts/build_traits.py to generate traits for a dataset's labels.json and save text protos
(using vanalens.encoders.Encoder.encode_text) to $VANALENS_CACHE/text/<dataset>.npz. Print 2 example species only.
Then write scripts/baseline_vlm.py (baseline B4): 100 test images (seed 0) from a test manifest, ask the model
open-ended "what plant species is this? answer with the Latin binomial", exact species match rate (and genus match)
-> runs/L4/b4_<dataset>.json. Cache raw answers under $VANALENS_CACHE/vlm/.
If VANALENS_LLM_KEY is empty: write the traits yourself (you are an LLM) for every species in each labels.json,
6 per species, into the same cache JSON format, skip B4, and note "reranker and B4 disabled: no key" in your DONE entry.
When done: DONE entry in worklog/L4.md, commit with prefix [L4], `git push origin HEAD:lane/L4-context`, stop.
```

## L5 app

**L5-a**
```
First write `L5` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L5-app` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L5-a to worklog/L5.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L5] start L5-a"`, `git push origin HEAD:lane/L5-app`) so the dispatcher sees the card is in progress.
You are lane L5 (app). Read AGENTS.md and vanalens/contracts.py. Implement app/app.py per its docstring with
Gradio, using only vanalens.pipeline.get_pipeline() and VANALENS_MOCK=1 during development. Mobile-friendly, no CDN.
Done when `pytest -q tests/test_app.py` passes and `VANALENS_MOCK=1 python -m app.app` serves the UI (PowerShell:
`$env:VANALENS_MOCK="1"; & $env:VANALENS_PY -m app.app`; start it in the background, check it answers on
http://127.0.0.1:7860, then stop it).
When done: DONE entry in worklog/L5.md, commit with prefix [L5], `git push origin HEAD:lane/L5-app`, stop.
```

**L5-b**
```
First write `L5` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L5-app` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L5-b to worklog/L5.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L5] start L5-b"`, `git push origin HEAD:lane/L5-app`) so the dispatcher sees the card is in progress.
Lane L5, card b. Create space/ with what a Hugging Face Space needs to run app/app.py (entry file, requirements,
README front matter). Default to VANALENS_MOCK=0 and load state from the cache path given by env. Do not deploy;
print the exact commands for the human.
When done: DONE entry in worklog/L5.md, commit with prefix [L5], `git push origin HEAD:lane/L5-app`, stop.
```

## L6 integration

**L6-m (merge; the dispatcher runs this itself in the MAIN checkout every run)**
```
You are the merge step, in the main checkout on branch main (there is no .lane file here).
Run `"$VANALENS_PY" scripts/merge_lanes.py` (PowerShell: `& $env:VANALENS_PY scripts/merge_lanes.py`). It merges
every lane with an unmerged DONE entry, one at a time: ownership check, `git merge --no-commit`, the lane's gate
tests (tooling + that lane + every lane already merged), commit or abort, then logs worklog/MERGE.md and pushes main.
Read its JSON. Do not merge by hand and do not edit any lane's code. Stop.
```

**L6-a**
```
First write `L6` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L6-integration` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L6-a to worklog/L6.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L6] start L6-a"`, `git push origin HEAD:lane/L6-integration`) so the dispatcher sees the card is in progress.
You are lane L6 (integration). Read AGENTS.md, vanalens/contracts.py and the docstring of vanalens/pipeline.py.
Implement VanaLens (from_cache, predict, add_species) by calling the other lanes' public functions only.
State comes from $VANALENS_CACHE/state/<dataset>.json (written by L3-b); fit the geo prior strength beta on the
cal split (small grid, NLL) and add it to that state file. Skip the text scorer when
$VANALENS_CACHE/text/<dataset>.npz is missing and the reranker when VANALENS_LLM_KEY is empty.
Write tests/test_pipeline.py using the synthetic fixtures (monkeypatch the encoder; no GPU, no network, no cache
needed). Change get_pipeline's default dataset only if inat_india is not available. Then run end-to-end on 20
inat_india test photos (and 20 campus photos if that manifest exists) and save runs/L6/e2e_<dataset>.json
(top-1, set size, coverage, seconds per photo). Keep MockPipeline unchanged.
When done: DONE entry in worklog/L6.md, commit with prefix [L6], `git push origin HEAD:lane/L6-integration`, stop.
```

**L6-b**
```
First write `L6` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L6-integration` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L6-b to worklog/L6.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L6] start L6-b"`, `git push origin HEAD:lane/L6-integration`) so the dispatcher sees the card is in progress.
Lane L6, card b. Do not merge anything (the dispatcher merges). Your branch already has main. Run the full test
suite (`pytest -q`, GPU tests skip). Then write scripts/final_table.py and build runs/L6/final.json: every baseline
that has a result file in runs/ (B0 runs/L1/b0_*, B1 runs/L2/b1_*, B2 runs/L2/b2_*, B3 = probe row, B4 runs/L4/b4_*)
and every ablation step (rows from runs/L3/ablation_*: knn, probe, text, fused, fused+T, fused+T+sets; then
compute "+ geo prior" and "+ reranker" yourself with VanaLens.predict on the test manifest, the reranker only if
VANALENS_LLM_KEY is set, at most 200 reranker calls) for each dataset that has a test manifest (inat_india always; dimpsar, campus if present), with n,
top-1, top-5, macro top-1, coverage and mean set size per row. Missing results become rows marked "not run".
Print the table as markdown, nothing else.
When done: DONE entry in worklog/L6.md, commit with prefix [L6], `git push origin HEAD:lane/L6-integration`, stop.
```

## L7 pitch

**L7-a**
```
First write `L7` to `.lane` (see AGENTS.md section 1), use the venv Python `$VANALENS_PY` for every python/pytest command, `git pull --no-rebase --no-edit origin lane/L7-pitch` and `git pull --no-rebase --no-edit origin main` (merge, never rebase), then add a STARTED entry for L7-a to worklog/L7.md (format: AGENTS.md section 6), and commit + push that entry right away (`git commit -m "[L7] start L7-a"`, `git push origin HEAD:lane/L7-pitch`) so the dispatcher sees the card is in progress.
You are lane L7 (pitch). Read AGENTS.md, docs/HANDOFF.md, docs/RESEARCH.md section 10 only, and runs/L6/final.json.
Write scripts/build_deck.py that builds pitch/VanaLens.pptx with python-pptx: 8 slides following the outline,
16:9, clean light theme, one idea per slide, charts rendered with matplotlib into pitch/*.png (results vs
baselines, ablation, few-shot curve if present). Also write pitch/script.md (speaker notes, 4 minutes total) and
pitch/demo.md (live demo steps on the Gradio app + fallback if Wi-Fi fails). Use only numbers that exist in runs/;
if a metric is missing, say so on the slide. No em dashes anywhere. Run the script and check the .pptx opens.
When done: DONE entry in worklog/L7.md, commit with prefix [L7], `git push origin HEAD:lane/L7-pitch`, stop.
```
