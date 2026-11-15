# Lanes and task cards

All work happens in ONE T3 project (the main checkout). One lane = one folder `.lanes/LX` inside it (a git worktree
created by `scripts/lane.py`) on its own branch = one agent thread at a time. The project root stays on main.
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
| Human / dispatcher | `AGENTS.md`, `CLAUDE.md`, `docs/*`, `vanalens/contracts.py`, `vanalens/config.py`, tooling scripts; merges via `scripts/merge_lanes.py` | | Claude (project root) | `tests/test_tooling.py` |

New tests go in new files you own; never edit someone else's test.

## Waves (maps onto the 24h plan)

| Hours | Run in parallel | Gate |
|---|---|---|
| 0-1 | Human: `docs/AUTOPILOT.md` (setup, one T3 project, schedule) | `pytest --co` collects 49 tests; `tests/test_tooling.py` 15 passed |
| 1-6 | Claude: L3-a. Antigravity A: L1-a + L2-a. Antigravity B: L4-a + L5-a | each lane's tests green |
| 3-6 | L1-b, then L2-b, L2-c (GPU), L1-c | inat_india manifests + embeddings cached; baseline JSONs in `runs/` |
| 6-9 | Claude: L3-b then L6-a. Antigravity B: L4-b | **MVP: real pipeline answers an inat_india test photo** |
| 9-16 | L6-b, L5-b | final table `runs/L6/final.json` |
| wave 2 | L1-d (vl600), then L2-e + L4-c, L5-c; then L3-c, L6-c, L7-b (deck); DINO fix L2-f, then L3-d, L6-d, in parallel with L7-c (report), L7-e (related work + novelty); then L7-d (review + number refresh) | runs/L6/final.json has vl600 with CIs; report + reviewed deck |
| 16-19 | fixes only | feature freeze at 19 |
| 19-24 | Claude: L7-a builds `pitch/VanaLens.pptx` from `runs/L6/final.json` | human rehearses |

---

## L1 data-eval

**L1-a**
```
Lane L1, card L1-a. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L1 --start L1-a` (bash: `"$VANALENS_PY" scripts/lane.py L1 --start L1-a`). It prepares your lane folder `.lanes/L1` (git worktree with the latest lane/L1-data-eval and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L1`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L1` (AGENTS.md section 1).
You are lane L1 (data-eval) on VanaLens. Read AGENTS.md, then vanalens/contracts.py and vanalens/config.py only.
Implement eval/metrics.py, eval/harness.py, and these functions in vanalens/data.py: grouped_split,
cap_per_class, write_manifest, read_manifest, read_exif_gps. Follow the docstrings exactly.
Done when `pytest -q tests/test_eval.py` passes. Do not implement the dataset loaders yet.
When done, inside `.lanes/L1`: DONE entry in worklog/L1.md, commit with prefix [L1], `git push origin HEAD:lane/L1-data-eval`, stop.
```

**L1-b**
```
Lane L1, card L1-b. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L1 --start L1-b` (bash: `"$VANALENS_PY" scripts/lane.py L1 --start L1-b`). It prepares your lane folder `.lanes/L1` (git worktree with the latest lane/L1-data-eval and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L1`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L1` (AGENTS.md section 1).
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
When done, inside `.lanes/L1`: DONE entry in worklog/L1.md, commit with prefix [L1], `git push origin HEAD:lane/L1-data-eval`, stop.
```

**L1-c**
```
Lane L1, card L1-c. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L1 --start L1-c` (bash: `"$VANALENS_PY" scripts/lane.py L1 --start L1-c`). It prepares your lane folder `.lanes/L1` (git worktree with the latest lane/L1-data-eval and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L1`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L1` (AGENTS.md section 1).
Lane L1, card c. Read AGENTS.md. Write scripts/baseline_plantnet_api.py (baseline B0): sample 400 images (seed 0)
from a test manifest, call the Pl@ntNet API v2 identify endpoint (read my.plantnet.org docs; key from env
PLANTNET_API_KEY; pass the organ when known), cache each raw JSON under $VANALENS_CACHE/plantnet_api/, map returned
species names to taxa idx (unmatched -> uniform row), and save results with eval.harness to runs/L1/b0_<dataset>.json.
Never exceed 450 calls per day; skip images already cached. If PLANTNET_API_KEY is empty, write the script,
skip the run, and mark the DONE entry 'B0 skipped: no key'.
When done, inside `.lanes/L1`: DONE entry in worklog/L1.md, commit with prefix [L1], `git push origin HEAD:lane/L1-data-eval`, stop.
```

**L1-d**
```
Lane L1, card L1-d. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L1 --start L1-d` (bash: `"$VANALENS_PY" scripts/lane.py L1 --start L1-d`). It prepares your lane folder `.lanes/L1` (git worktree with the latest lane/L1-data-eval and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L1`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L1` (AGENTS.md section 1).
You are lane L1 (data-eval). Read AGENTS.md, vanalens/contracts.py, vanalens/config.py. Build dataset "vl600"
(automatic, resumable). Builder in eval/build_vl600.py (run `python -m eval.build_vl600`), loader load_vl600 in
vanalens/data.py, manifests via scripts/make_manifests.py.
1. Species, selected by observation count ONLY (never drop a species later for accuracy). (a) the ~400 most observed
   plant species in India: iNat API v1 observations/species_counts, place = India, taxon_id 47126, rank species.
   (b) the ~200 most observed plant species worldwide (same endpoint, no place) not already in (a): houseplants,
   crops, iconic trees and flowers. (c) always add this common-Indian list, resolved to accepted iNat binomials (use
   these unless iNat says otherwise; where a name is a genus take its most observed species in India): tulsi Ocimum
   tenuiflorum, neem Azadirachta indica, hibiscus Hibiscus rosa-sinensis, rose (most observed Rosa), money plant
   Epipremnum aureum, bougainvillea (most observed Bougainvillea), marigold Tagetes erecta, mango Mangifera indica,
   peepal Ficus religiosa, banyan Ficus benghalensis, curry leaf Murraya koenigii, aloe vera Aloe vera, guava Psidium
   guajava, papaya Carica papaya, banana (most observed Musa), coconut Cocos nucifera, jasmine Jasminum sambac,
   oleander Nerium oleander, ixora Ixora coccinea, plumeria Plumeria rubra, gulmohar Delonix regia, amaltas Cassia
   fistula, ashoka Saraca asoca and Monoon longifolium, teak Tectona grandis, moringa Moringa oleifera, tamarind
   Tamarindus indica, jamun Syzygium cumini, lantana Lantana camara, parthenium Parthenium hysterophorus, calotropis
   Calotropis procera and Calotropis gigantea, datura (most observed Datura), castor Ricinus communis, mimosa Mimosa
   pudica, snake plant Dracaena trifasciata, spider plant Chlorophytum comosum, crown of thorns Euphorbia milii,
   periwinkle Catharanthus roseus, croton Codiaeum variegatum, areca palm Dypsis lutescens, bamboo (most observed
   bamboo species in India, e.g. Bambusa vulgaris). All growth forms stay (trees, shrubs, herbs, grasses, climbers,
   ferns, succulents, aquatic, crops). Save $VANALENS_DATA/vl600/species.csv (name, common name, source
   india/world/common, observation counts).
2. Photos, per species: observations with a CC0 / CC-BY / CC-BY-NC photo, research grade; list (c) may also use
   casual or needs_id observations (cultivated allowed) whose community taxon is the species with >=1 agreeing ID.
   Up to 80 photos observed before 2025-07-01 (gallery + cal) and up to 15 observations on or after 2025-07-01
   (test), at most 3 photos per observation; prefer observations in India, fill from worldwide. Medium size.
   API list calls at most 1 request/second with a User-Agent. Photo files ONLY from the iNaturalist AWS open data
   bucket: https://inaturalist-open-data.s3.amazonaws.com/photos/<photo_id>/medium.<ext>, with 8 parallel download
   workers (the 1 request/second limit is for API calls only) and retries with backoff. Save
   $VANALENS_DATA/vl600/<Genus species>/<obs_id>_<photo_id>.jpg and meta.csv
   (image,obs_id,photo_id,lat,lon,observed_on,license,url,quality_grade,in_india).
   Resumable (skip existing files, checkpoint API progress); run in the background with output to
   runs/logs/build_vl600.log and read only its last 30 lines. Before downloading, write the planned photo count and
   expected GB (about 80 KB per medium photo) to the log. Disk space is not a constraint (human decision); stop as
   BLOCKED only if the disk would drop under 10 GB free.
3. Splits: group = obs_id so an observation's photos stay together and never cross splits; test by the date rule;
   cal = 20% of pre-cutoff observations by group; cap gallery with cap_per_class. Species with few or no test
   observations stay in (never drop); report how many have fewer than 5.
4. Coverage: share of all research-grade Indian plant observations (species_counts total, place India, Plantae)
   covered by the chosen species. Save runs/L1/vl600_stats.json (species, photos and observations per split, GB,
   coverage %, species with <5 test observations).
5. campus: if $VANALENS_DATA/campus exists (one folder per species, human phone photos), implement load_campus and
   write ONLY a campus_test manifest (labels mapped onto vl600; species not in vl600 kept and listed for open-set).
   Otherwise skip and say so.
Add offline tests (no network) in tests/test_l1_vl600.py; keep tests/test_eval.py green.
DONE entry Results: species, photos, GB, coverage %.
When done, inside `.lanes/L1`: DONE entry in worklog/L1.md, commit with prefix [L1], `git push origin HEAD:lane/L1-data-eval`, stop.
```

## L2 encoders (GPU)

**L2-a**
```
Lane L2, card L2-a. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L2 --start L2-a` (bash: `"$VANALENS_PY" scripts/lane.py L2 --start L2-a`). It prepares your lane folder `.lanes/L2` (git worktree with the latest lane/L2-encoders and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L2`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L2` (AGENTS.md section 1).
You are lane L2 (encoders), the only lane allowed to use the GPU. Read AGENTS.md, vanalens/contracts.py,
vanalens/config.py. Implement vanalens/encoders.py per its docstring (BioCLIP 2 via open_clip + DINOv3 via
transformers, fp16, 6 GB GPU). If DINOv3 access is still pending, use the documented dinov2-large fallback.
Done when `VANALENS_GPU_TESTS=1 pytest -q tests/test_encoders.py` passes (PowerShell:
`$env:VANALENS_GPU_TESTS="1"; & $env:VANALENS_PY -m pytest -q tests/test_encoders.py`). Report images/second.
When done, inside `.lanes/L2`: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

**L2-b**
```
Lane L2, card L2-b. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L2 --start L2-b` (bash: `"$VANALENS_PY" scripts/lane.py L2 --start L2-b`). It prepares your lane folder `.lanes/L2` (git worktree with the latest lane/L2-encoders and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L2`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L2` (AGENTS.md section 1).
Lane L2, card b. Write scripts/baseline_bioclip.py (baseline B2): pybioclip CustomLabelsClassifier over the
dataset's species names on a test manifest -> (N, C) matrix in taxa order -> eval.harness ->
runs/L2/b2_<dataset>.json. Also record the open-ended TreeOfLifeClassifier top-1/top-5 species-name match rate.
When done, inside `.lanes/L2`: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

**L2-c**
```
Lane L2, card L2-c. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L2 --start L2-c` (bash: `"$VANALENS_PY" scripts/lane.py L2 --start L2-c`). It prepares your lane folder `.lanes/L2` (git worktree with the latest lane/L2-encoders and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L2`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L2` (AGENTS.md section 1).
Lane L2, card c. Write scripts/extract_all.py (cross-platform, stdlib + vanalens): for every manifest in
$VANALENS_CACHE/manifests/ without an up-to-date $VANALENS_CACHE/emb/<same name>.npz, run
`sys.executable -m vanalens.encoders --manifest ... --out ...` one after another, logging to
runs/logs/extract_all.log and printing "ALL DONE" at the end. `--detach` relaunches itself in the background
(Windows: subprocess creationflags DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP; POSIX: start_new_session=True).
Start it with --detach, then check the last 5 log lines every few minutes until "ALL DONE". Write the DONE entry
only when every manifest has its .npz (check with a one-line python -c that prints names and shapes).
When done, inside `.lanes/L2`: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

**L2-d (optional)**
```
Lane L2, card L2-d. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L2 --start L2-d` (bash: `"$VANALENS_PY" scripts/lane.py L2 --start L2-d`). It prepares your lane folder `.lanes/L2` (git worktree with the latest lane/L2-encoders and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L2`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L2` (AGENTS.md section 1).
Lane L2, card d. Write scripts/baseline_plantnet300k.py (baseline B1): load the pretrained ResNet weights from the
plantnet/PlantNet-300K repo, evaluate on the Pl@ntNet-300K test manifest, save runs/L2/b1_plantnet300k.json.
When done, inside `.lanes/L2`: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

**L2-e**
```
Lane L2, card L2-e. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L2 --start L2-e` (bash: `"$VANALENS_PY" scripts/lane.py L2 --start L2-e`). It prepares your lane folder `.lanes/L2` (git worktree with the latest lane/L2-encoders and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L2`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L2` (AGENTS.md section 1).
You are lane L2 (encoders, the only GPU lane). Read AGENTS.md. Embed vl600 (gallery, cal, test) and campus_test if
that manifest exists with scripts/extract_all.py, in the background with output to runs/logs/extract_vl600.log
(resume-safe; read only the last 30 lines). Make only the changes extract_all.py needs for ~60k images (sharding,
batch size that fits 6 GB, skip and log unreadable images). Accuracy target is 95%+ (human), so also do test-time
augmentation: for every image store the mean of the embeddings of the original and its horizontal flip (each block
re-normalised) as X, and for the gallery split also save 2 extra augmented views per image (random resized crop
0.6-1.0 + flip, seed 0) as X_aug/y_aug in the same npz for probe training. Compute and record in the DONE entry the
probe top-1 on cal with and without TTA (quick sklearn check) so L3 knows it helps; if TTA hurts, keep plain X.
Then run the B2 BioCLIP 2 zero-shot baseline on the vl600
test split (scripts/baseline_bioclip.py) -> runs/L2/b2_vl600.json. Keep inat_india caches untouched.
Lane tests must pass. DONE entry Results: shapes per split, images/s, failed images, B2 top-1/top-5.
When done, inside `.lanes/L2`: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

**L2-f**
```
Lane L2, card L2-f. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L2 --start L2-f` (bash: `"$VANALENS_PY" scripts/lane.py L2 --start L2-f`). It prepares your lane folder `.lanes/L2` (git worktree with the latest lane/L2-encoders and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L2`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L2` (AGENTS.md section 1).
You are lane L2 (encoders, the only GPU lane). Read AGENTS.md. URGENT BUG (found by the lead, 19:50): the DINO block
of every cached embedding is noise. DINOv3 is gated (401, no HF login), so Encoder falls back to facebook/dinov2-large,
but `_load_dino` builds the model with AutoModel.from_config and fills it with `_stream_safetensors_into_model`, which
matches tensors by name: the installed transformers names DINOv2 attention `attention.q_proj/k_proj/v_proj/o_proj`
while the checkpoint uses the old names, so 192 of 439 tensors (all attention weights) are silently skipped and stay
randomly initialised, different on every load. Proof: re-encoding vl600 gallery images gives BioCLIP cos 1.00 vs the
cache but DINO cos only 0.3-0.58 (same GPU, fp16, TTA). This also breaks the live app (open-set rejects ~95%).
Do: (1) fix DINO loading so every model parameter is loaded from the checkpoint (e.g. AutoModel.from_pretrained with
dtype, or a correct key remap), and make the streaming loader raise if any model parameter/buffer is left unloaded
(also check the BioCLIP path). (2) New test tests/test_l2_weights.py: loading the DINO model leaves 0 unloaded
tensors, and two Encoder loads give identical embeddings for the same image. (3) Recompute the DINO block for ALL
cached splits in $VANALENS_CACHE/emb: vl600 gallery (X, X_plain, X_aug with the same crop/flip views, seed 0), cal,
test, and inat_india gallery/cal/test. Prefer recomputing only the DINO block and keeping the BioCLIP block
bit-identical (faster); a full re-extract is fine if simpler. Keep flip TTA and per-block re-normalisation exactly as
before. Set the npz `encoder` field to the real model name (e.g. "bioclip2+dinov2-large"). Write to temp files and
swap atomically so a crash never leaves a broken cache; background job, log to runs/logs/fix_dino.log, read only the
last 30 lines. GPU: L7 may have app/app.py open (about 1.5 GB VRAM); pick a batch size that fits next to it.
(4) Verify: re-encode 20 random gallery images live and report min cos vs cache for both blocks (must be > 0.99), and
the quick sklearn probe top-1 on vl600 cal (was 0.9051 with TTA) with the new embeddings.
DONE entry Results: unloaded-tensor count before/after, live-vs-cache cos, probe cal top-1 before/after, images/s.
When done, inside `.lanes/L2`: DONE entry in worklog/L2.md, commit with prefix [L2], `git push origin HEAD:lane/L2-encoders`, stop.
```

## L3 core-ml

**L3-a**
```
Lane L3, card L3-a. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L3 --start L3-a` (bash: `"$VANALENS_PY" scripts/lane.py L3 --start L3-a`). It prepares your lane folder `.lanes/L3` (git worktree with the latest lane/L3-core-ml and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L3`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L3` (AGENTS.md section 1).
You are lane L3 (core-ml). Read AGENTS.md, vanalens/contracts.py, vanalens/config.py. Implement
vanalens/scorers.py, vanalens/fusion.py, vanalens/conformal.py and vanalens/hierarchy.py exactly as their
docstrings say (KNNScorer uses faiss.IndexFlatIP). No data, no GPU: use the synthetic fixtures.
Done when `pytest -q tests/test_core.py` passes.
When done, inside `.lanes/L3`: DONE entry in worklog/L3.md, commit with prefix [L3], `git push origin HEAD:lane/L3-core-ml`, stop.
```

**L3-b**
```
Lane L3, card L3-b. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L3 --start L3-b` (bash: `"$VANALENS_PY" scripts/lane.py L3 --start L3-b`). It prepares your lane folder `.lanes/L3` (git worktree with the latest lane/L3-core-ml and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L3`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L3` (AGENTS.md section 1).
Lane L3, card b. Write scripts/fit_core.py --dataset <name>: load gallery/cal/test npz from $VANALENS_CACHE/emb and
labels.json; fit ProbeScorer on gallery; compute kNN/probe(/text if protos exist) on cal; fit_weights + temperature
on cal; fit qhat (alpha from config) and an open-set threshold on cal; save
$VANALENS_CACHE/state/<dataset>.json; evaluate on test and write runs/L3/ablation_<dataset>.json with rows:
knn, probe, text, fused, fused+T, fused+T+sets. Print only the summary table.
When done, inside `.lanes/L3`: DONE entry in worklog/L3.md, commit with prefix [L3], `git push origin HEAD:lane/L3-core-ml`, stop.
```

**L3-c**
```
Lane L3, card L3-c. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L3 --start L3-c` (bash: `"$VANALENS_PY" scripts/lane.py L3 --start L3-c`). It prepares your lane folder `.lanes/L3` (git worktree with the latest lane/L3-core-ml and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L3`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L3` (AGENTS.md section 1).
You are lane L3 (core-ml). Read AGENTS.md, vanalens/contracts.py, vanalens/config.py. Fit the core on vl600 with
scripts/fit_core.py --dataset vl600 (inat_india must keep working). For this card only L3 may use the GPU (L2 has no
GPU work queued after L2-e); first check nvidia-smi shows no other python job, otherwise use CPU.
Probe: a torch linear layer on the cached embeddings, weight decay tuned on cal (sklearn CPU fallback). kNN as
before. Text scorer from $VANALENS_CACHE/text/vl600.npz. Fusion weights, temperature, split conformal (alpha 0.1)
and open-set threshold are all fit on cal only. Ablation runs/L3/ablation_vl600.json: rows knn, probe, text, fused,
fused+T, fused+T+sets and mlp (one hidden layer of 1024, dropout, early stopping on cal), each with top-1, top-5,
macro top-1, NLL, ECE, coverage, mean set size. Keep the linear probe as the default unless the MLP beats it on cal by
more than 1 point top-1; record the choice in $VANALENS_CACHE/state/vl600.json.
Accuracy target is 95%+ top-1 (human). Push it honestly, choosing everything on cal only and never dropping species:
train the probe/MLP on gallery X plus X_aug if present; tune weight decay, label smoothing (0, 0.1) and class-balanced
weights; try an ensemble (probe + MLP + kNN + text, weights fit on cal) and keep it if it wins on cal; add rows
"ensemble" and "mlp" to the ablation. Report test top-1, top-5, genus and family top-1. If 95% single-photo top-1 is
not reached, say so plainly with the number; do not tune on test.
Add synthetic CPU tests in tests/test_l3_vl600.py; tests/test_core.py stays green.
When done, inside `.lanes/L3`: DONE entry in worklog/L3.md, commit with prefix [L3], `git push origin HEAD:lane/L3-core-ml`, stop.
```

**L3-d**
```
Lane L3, card L3-d. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L3 --start L3-d` (bash: `"$VANALENS_PY" scripts/lane.py L3 --start L3-d`). It prepares your lane folder `.lanes/L3` (git worktree with the latest lane/L3-core-ml and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L3`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L3` (AGENTS.md section 1).
You are lane L3 (core-ml). Read AGENTS.md and the L2-f DONE entry in scripts/status.py output. L2-f replaced the DINO
block of every cached embedding (before it, the DINO half was random noise from a weight-loading bug), so every fitted
state from L3-c is stale. Re-run scripts/fit_core.py for vl600 and inat_india on the new embeddings with the same
procedure as L3-c: head (linear / MLP) hyper-parameters, kNN, text weight, ensemble weights, temperature, conformal
qhat and open-set threshold all re-chosen on cal only, never on test. Also re-check per-block weighting now that the
DINO block carries signal (e.g. a block weight for DINO vs BioCLIP chosen on cal) and keep it only if cal improves.
Update runs/L3/ablation_{vl600,inat_india}.json. Lane tests must pass. DONE entry Results: vl600 test top-1/top-5,
genus, family, cal top-1, before (L3-c: test top-1 .933, top-5 .979) vs after; inat_india ensemble top-1.
When done, inside `.lanes/L3`: DONE entry in worklog/L3.md, commit with prefix [L3], `git push origin HEAD:lane/L3-core-ml`, stop.
```

## L4 context

**L4-a**
```
Lane L4, card L4-a. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L4 --start L4-a` (bash: `"$VANALENS_PY" scripts/lane.py L4 --start L4-a`). It prepares your lane folder `.lanes/L4` (git worktree with the latest lane/L4-context and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L4`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L4` (AGENTS.md section 1).
You are lane L4 (context). Read AGENTS.md, vanalens/contracts.py, vanalens/config.py. Implement the offline
functions: geo_prior and apply_prior (vanalens/geo.py); trait_prompts and build_text_protos (vanalens/traits.py);
should_rerank, build_rerank_prompt, parse_choice (vanalens/rerank.py). Done when `pytest -q tests/test_context.py`
passes. No network calls in this card.
When done, inside `.lanes/L4`: DONE entry in worklog/L4.md, commit with prefix [L4], `git push origin HEAD:lane/L4-context`, stop.
```

**L4-b**
```
Lane L4, card L4-b. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L4 --start L4-b` (bash: `"$VANALENS_PY" scripts/lane.py L4 --start L4-b`). It prepares your lane folder `.lanes/L4` (git worktree with the latest lane/L4-context and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L4`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L4` (AGENTS.md section 1).
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
When done, inside `.lanes/L4`: DONE entry in worklog/L4.md, commit with prefix [L4], `git push origin HEAD:lane/L4-context`, stop.
```

**L4-c**
```
Lane L4, card L4-c. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L4 --start L4-c` (bash: `"$VANALENS_PY" scripts/lane.py L4 --start L4-c`). It prepares your lane folder `.lanes/L4` (git worktree with the latest lane/L4-context and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L4`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L4` (AGENTS.md section 1).
You are lane L4 (context). Read AGENTS.md, vanalens/contracts.py. Build text prototypes for EVERY species in
$VANALENS_DATA/vl600/labels.json with the same method as inat_india (scripts/build_traits.py, vanalens/traits.py)
-> $VANALENS_CACHE/text/vl600.npz. VANALENS_LLM_KEY is empty: write the traits yourself in batches of 50 species,
saving after each batch (resumable). If a species still has no traits, use a taxonomic prompt
("a photo of <family> <genus> <species>, <common name>") and list those species. No species may be missing from the
npz. Also pre-fetch the GBIF geo cells the vl600 test photos with GPS need (cache under $VANALENS_CACHE/gbif).
Add offline tests in tests/test_l4_vl600.py; tests/test_context.py stays green.
DONE entry Results: species with traits, taxonomic-fallback count, GBIF cells cached.
When done, inside `.lanes/L4`: DONE entry in worklog/L4.md, commit with prefix [L4], `git push origin HEAD:lane/L4-context`, stop.
```

## L5 app

**L5-a**
```
Lane L5, card L5-a. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L5 --start L5-a` (bash: `"$VANALENS_PY" scripts/lane.py L5 --start L5-a`). It prepares your lane folder `.lanes/L5` (git worktree with the latest lane/L5-app and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L5`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L5` (AGENTS.md section 1).
You are lane L5 (app). Read AGENTS.md and vanalens/contracts.py. Implement app/app.py per its docstring with
Gradio, using only vanalens.pipeline.get_pipeline() and VANALENS_MOCK=1 during development. Mobile-friendly, no CDN.
Done when `pytest -q tests/test_app.py` passes and `VANALENS_MOCK=1 python -m app.app` serves the UI (PowerShell:
`$env:VANALENS_MOCK="1"; & $env:VANALENS_PY -m app.app`; start it in the background, check it answers on
http://127.0.0.1:7860, then stop it).
When done, inside `.lanes/L5`: DONE entry in worklog/L5.md, commit with prefix [L5], `git push origin HEAD:lane/L5-app`, stop.
```

**L5-b**
```
Lane L5, card L5-b. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L5 --start L5-b` (bash: `"$VANALENS_PY" scripts/lane.py L5 --start L5-b`). It prepares your lane folder `.lanes/L5` (git worktree with the latest lane/L5-app and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L5`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L5` (AGENTS.md section 1).
Lane L5, card b. Create space/ with what a Hugging Face Space needs to run app/app.py (entry file, requirements,
README front matter). Default to VANALENS_MOCK=0 and load state from the cache path given by env. Do not deploy;
print the exact commands for the human.
When done, inside `.lanes/L5`: DONE entry in worklog/L5.md, commit with prefix [L5], `git push origin HEAD:lane/L5-app`, stop.
```

**L5-c**
```
Lane L5, card L5-c. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L5 --start L5-c` (bash: `"$VANALENS_PY" scripts/lane.py L5 --start L5-c`). It prepares your lane folder `.lanes/L5` (git worktree with the latest lane/L5-app and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L5`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L5` (AGENTS.md section 1).
You are lane L5 (app). Read AGENTS.md, vanalens/contracts.py and app/. Extend the Gradio app:
1. Accept 1 to 3 photos of the same plant and send them as one observation to the pipeline (pass the list to
   VanaLens.predict; if the pipeline only takes one photo, average the calibrated log-probabilities in an app helper
   and say so in your DONE entry for L6).
2. When not confident (conformal set has more than one species, or top probability below the state threshold), show
   "Not sure: A, B, C" with the set members and their probabilities.
3. When the open-set reject fires, show "Not in catalog, closest genus/family X" with an "Add this species" option
   (name + up to 10 photos, calls add_species, no retraining).
Keep MockPipeline paths working so tests run without caches; update tests/test_app.py; mirror the change in space/
if it duplicates the app.
When done, inside `.lanes/L5`: DONE entry in worklog/L5.md, commit with prefix [L5], `git push origin HEAD:lane/L5-app`, stop.
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
Lane L6, card L6-a. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L6 --start L6-a` (bash: `"$VANALENS_PY" scripts/lane.py L6 --start L6-a`). It prepares your lane folder `.lanes/L6` (git worktree with the latest lane/L6-integration and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L6`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L6` (AGENTS.md section 1).
You are lane L6 (integration). Read AGENTS.md, vanalens/contracts.py and the docstring of vanalens/pipeline.py.
Implement VanaLens (from_cache, predict, add_species) by calling the other lanes' public functions only.
State comes from $VANALENS_CACHE/state/<dataset>.json (written by L3-b); fit the geo prior strength beta on the
cal split (small grid, NLL) and add it to that state file. Skip the text scorer when
$VANALENS_CACHE/text/<dataset>.npz is missing and the reranker when VANALENS_LLM_KEY is empty.
Write tests/test_pipeline.py using the synthetic fixtures (monkeypatch the encoder; no GPU, no network, no cache
needed). Change get_pipeline's default dataset only if inat_india is not available. Then run end-to-end on 20
inat_india test photos (and 20 campus photos if that manifest exists) and save runs/L6/e2e_<dataset>.json
(top-1, set size, coverage, seconds per photo). Keep MockPipeline unchanged.
When done, inside `.lanes/L6`: DONE entry in worklog/L6.md, commit with prefix [L6], `git push origin HEAD:lane/L6-integration`, stop.
```

**L6-b**
```
Lane L6, card L6-b. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L6 --start L6-b` (bash: `"$VANALENS_PY" scripts/lane.py L6 --start L6-b`). It prepares your lane folder `.lanes/L6` (git worktree with the latest lane/L6-integration and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L6`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L6` (AGENTS.md section 1).
Lane L6, card b. Do not merge anything (the dispatcher merges). Your branch already has main. Run the full test
suite (`pytest -q`, GPU tests skip). Then write scripts/final_table.py and build runs/L6/final.json: every baseline
that has a result file in runs/ (B0 runs/L1/b0_*, B1 runs/L2/b1_*, B2 runs/L2/b2_*, B3 = probe row, B4 runs/L4/b4_*)
and every ablation step (rows from runs/L3/ablation_*: knn, probe, text, fused, fused+T, fused+T+sets; then
compute "+ geo prior" and "+ reranker" yourself with VanaLens.predict on the test manifest, the reranker only if
VANALENS_LLM_KEY is set, at most 200 reranker calls) for each dataset that has a test manifest (inat_india always; dimpsar, campus if present), with n,
top-1, top-5, macro top-1, coverage and mean set size per row. Missing results become rows marked "not run".
Print the table as markdown, nothing else.
When done, inside `.lanes/L6`: DONE entry in worklog/L6.md, commit with prefix [L6], `git push origin HEAD:lane/L6-integration`, stop.
```

**L6-c**
```
Lane L6, card L6-c. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L6 --start L6-c` (bash: `"$VANALENS_PY" scripts/lane.py L6 --start L6-c`). It prepares your lane folder `.lanes/L6` (git worktree with the latest lane/L6-integration and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L6`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L6` (AGENTS.md section 1).
You are lane L6 (integration). Read AGENTS.md, vanalens/contracts.py, vanalens/pipeline.py. CPU only (cached
embeddings).
1. Multi-photo fusion: VanaLens.predict on an observation (list of up to 3 photos) averages the calibrated
   log-probabilities of its photos, renormalises, then applies the geo prior, conformal set and backoff.
2. Geo prior for vl600: fit beta on the vl600 cal split as in L6-a and store it in the vl600 state file.
3. Extend scripts/final_table.py -> runs/L6/final.json for inat_india, vl600 and campus (if its manifest exists),
   every number with a 95% bootstrap CI (1000 resamples by observation, seed 0):
   top-1 and top-5 for 1 photo and for up to 3 photos per plant; conformal coverage % and mean set size;
   selective accuracy (confidence threshold chosen on cal to reach >=95% accuracy on answered photos; report test
   accuracy on answered photos and the share answered); open-set (hold out 10% of species, seed 0, refit on the rest;
   reject rate on held-out species, false-reject rate on known species, AUROC); few-shot add-species curve (for the
   held-out species add 1, 3, 5, 10 gallery photos via add_species, top-1 on their test photos); baseline rows from
   runs/ as in L6-b (B2 vl600 from runs/L2/b2_vl600.json); ablation rows from runs/L3/ablation_vl600.json.
   Missing results become rows marked "not run".
4. End to end on 20 vl600 test observations -> runs/L6/e2e_vl600.json.
5. Accuracy target 95%+ (human). Report clearly which numbers reach it: single-photo top-1, 3-photo top-1, top-5,
   genus/family top-1 (backoff), and selective accuracy with its answered share. Never drop species, never tune on test.
Synthetic tests in tests/test_pipeline.py (no GPU, no network). Print the table as markdown, nothing else.
When done, inside `.lanes/L6`: DONE entry in worklog/L6.md, commit with prefix [L6], `git push origin HEAD:lane/L6-integration`, stop.
```

**L6-d**
```
Lane L6, card L6-d. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L6 --start L6-d` (bash: `"$VANALENS_PY" scripts/lane.py L6 --start L6-d`). It prepares your lane folder `.lanes/L6` (git worktree with the latest lane/L6-integration and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L6`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L6` (AGENTS.md section 1).
You are lane L6 (integration). Read AGENTS.md and the L2-f and L3-d DONE entries in scripts/status.py output. L2-f
fixed a DINO weight-loading bug (the DINO half of all embeddings was random and differed between the cache and the
live encoder) and L3-d refitted every state. Re-run scripts/final_table.py -> runs/L6/final.json and the live end-to-end
check -> runs/L6/e2e_vl600.json with the same procedure as L6-c (CIs by observation, 1 vs 3 photos, selective
accuracy, open-set holdout, few-shot curve, geo prior beta refit on cal). Confirm that the live pipeline now matches the
cache: e2e top-1 close to the cached test number and the unknown/reject rate on known test photos back to a normal
level (it was ~95% because of the bug); if live and cache still disagree, find out why and fix it in your files or
write BLOCKED with the exact cause. Also add a "baselines" block to final.json for vl600 test (same CIs by observation), all trained on gallery,
tuned on cal only: B2 BioCLIP 2 zero-shot (runs/L2), BioCLIP-only linear probe, DINOv2-only linear probe,
kNN-only, concatenated-feature probe without fusion, text-prototype-only (zero images), full VanaLens 1 photo
and 3 photos. These feed the report's head-to-head novelty table (L7-e/L7-d). Lane tests must pass. DONE entry Results: the same metric list as L6-c with
before -> after, and which metrics reach 95%.
When done, inside `.lanes/L6`: DONE entry in worklog/L6.md, commit with prefix [L6], `git push origin HEAD:lane/L6-integration`, stop.
```

## L7 pitch

**L7-a**
```
Lane L7, card L7-a. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L7 --start L7-a` (bash: `"$VANALENS_PY" scripts/lane.py L7 --start L7-a`). It prepares your lane folder `.lanes/L7` (git worktree with the latest lane/L7-pitch and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L7`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L7` (AGENTS.md section 1).
You are lane L7 (pitch). Read AGENTS.md, docs/HANDOFF.md, docs/RESEARCH.md section 10 only, and runs/L6/final.json.
Write scripts/build_deck.py that builds pitch/VanaLens.pptx with python-pptx: 8 slides following the outline,
16:9, clean light theme, one idea per slide, charts rendered with matplotlib into pitch/*.png (results vs
baselines, ablation, few-shot curve if present). Also write pitch/script.md (speaker notes, 4 minutes total) and
pitch/demo.md (live demo steps on the Gradio app + fallback if Wi-Fi fails). Use only numbers that exist in runs/;
if a metric is missing, say so on the slide. No em dashes anywhere. Run the script and check the .pptx opens.
When done, inside `.lanes/L7`: DONE entry in worklog/L7.md, commit with prefix [L7], `git push origin HEAD:lane/L7-pitch`, stop.
```

**L7-b**
```
Lane L7, card L7-b. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L7 --start L7-b` (bash: `"$VANALENS_PY" scripts/lane.py L7 --start L7-b`). It prepares your lane folder `.lanes/L7` (git worktree with the latest lane/L7-pitch and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L7`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L7` (AGENTS.md section 1).
You are lane L7 (pitch). Read AGENTS.md, docs/HANDOFF.md, docs/RESEARCH.md, runs/L6/final.json and the JSON files in
runs/. The current deck was judged weak by the human: REDESIGN pitch/VanaLens.pptx from scratch with
scripts/build_deck.py (python-pptx), simple but clean.
Theme: 16:9, white background, near-black text #1F2933, one accent green #2E7D32 plus light grey #E5E7EB for shapes,
one font family (Segoe UI; titles 32-36 pt bold, body 18-22 pt), generous margins (0.6 in), left-aligned titles,
small slide number bottom-right, no clip art, no gradients, no shadows, no word art. At most 4 bullets per slide and
about 12 words per bullet; one message per slide stated in the title as a sentence. Charts from matplotlib in the same
font and colours (accent for VanaLens, grey for baselines, no top/right spines, large labels, value labels on bars,
CI error bars, 200 dpi PNG in pitch/). Use a few real vl600 photos (CC-licensed; small attribution line on the slide).
Outline (11-13 slides): title; the problem (people want to name common plants, apps are overconfident); what we built
(one line + 3 key ideas); how it works (simple pipeline diagram drawn with shapes: photo -> BioCLIP 2 + DINOv3
embeddings -> probe / kNN / text scorers -> fusion + calibration -> location prior -> answer or "Not sure" set or
"Not in catalog"); dataset vl600 (607 species, 61,367 photos, India + common cultivated + global, 71.6% coverage,
test photos taken after the model's training cutoff); results vs baselines with 95% CIs; 1 photo vs 3 photos and
selective accuracy; honest uncertainty (calibration / conformal sets); unknown plants + add a species with 5 photos
(open-set and few-shot curve); app screenshots; limitations and next steps; thank you. Speaker notes on every slide.
Verify visually: export every slide to PNG with PowerPoint COM automation (PowerShell, Office is installed; e.g.
pitch/preview/slide_NN.png) and look at each image; fix any overflow, overlap, tiny text or empty areas, then
re-export. Also write pitch/script.md (4-5 minute talk). Use only numbers that exist in runs/. No em dashes.
When done, inside `.lanes/L7`: DONE entry in worklog/L7.md, commit with prefix [L7], `git push origin HEAD:lane/L7-pitch`, stop.
```

**L7-c**
```
Lane L7, card L7-c. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L7 --start L7-c` (bash: `"$VANALENS_PY" scripts/lane.py L7 --start L7-c`). It prepares your lane folder `.lanes/L7` (git worktree with the latest lane/L7-pitch and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L7`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L7` (AGENTS.md section 1).
You are lane L7 (pitch). Read AGENTS.md, docs/HANDOFF.md, docs/RESEARCH.md, runs/L6/final.json, the JSON files in
runs/, and the code docstrings you need (vanalens/*.py, eval/, scripts/) to describe the system accurately. Write the
PROJECT REPORT for submission: pitch/report/VanaLens_report.md, built into pitch/report/VanaLens_report.pdf
(10-16 pages) by pitch/report/build_report.py: markdown -> styled HTML (simple serif body, numbered headings,
figure captions, tables) -> PDF with headless Edge (`msedge --headless --print-to-pdf=...`, installed) or Word COM.
Required sections: 1 Introduction (problem, who it is for, goals); 2 How the system works (plain-language walkthrough
of one prediction, with the pipeline figure); 3 Approach and methods (frozen encoders BioCLIP 2 + DINOv3, linear
probe / MLP, kNN, text-trait prototypes, log-linear fusion, temperature scaling, split conformal sets, open-set reject,
hierarchical backoff, GBIF location prior, multi-photo fusion, add-species without retraining, and why no
fine-tuning); 4 Dataset (vl600 selection rules, splits by observation, post-cutoff test, licences, coverage; also
inat_india; dataset link: iNaturalist + the open-data bucket + how to rebuild); 5 Experimental setup (metrics, CIs,
baselines); 6 Results (tables with 95% CIs, ablation, 1 vs 3 photos, selective accuracy, calibration, open-set,
few-shot, per-growth-form or hardest-species error analysis if the data allows); 7 Limitations of the current
implementation (specific and honest); 8 How we plan to overcome them and improve (one concrete step per limitation,
plus future features such as offline mobile app, more regional species, expert feedback loop); 9 Conclusion;
References. Figures from pitch/*.png; add a confusion example figure if helpful. Use only numbers that exist in
runs/; where a result is missing say "not measured". L2-f/L3-d/L6-d (DINO weight-loading fix, see
docs/HANDOFF.md 19:55) are re-running in parallel and will change the numbers: write the report now from the current
runs/ files, but make every result number come from a small generated include (e.g. build_report.py reads runs/L6/final.json
and runs/L3/ablation_*.json and fills the tables), so L7-d can refresh it with one rebuild; describe the bug and its
fix briefly and honestly (methods or limitations). Write clearly and concretely. No em dashes.
When done, inside `.lanes/L7`: DONE entry in worklog/L7.md, commit with prefix [L7], `git push origin HEAD:lane/L7-pitch`, stop.
```

**L7-e**
```
Lane L7, card L7-e. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L7 --start L7-e` (bash: `"$VANALENS_PY" scripts/lane.py L7 --start L7-e`). It prepares your lane folder `.lanes/L7` (git worktree with the latest lane/L7-pitch and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L7`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L7` (AGENTS.md section 1).
You are lane L7 (pitch). Read AGENTS.md, docs/RESEARCH.md (sections 3-6: datasets, baselines, research summary,
recommended system), pitch/report/VanaLens_report.md, runs/L2/b2_vl600.json, runs/L3/ablation_*.json, runs/L6/final.json.
The human says NOVELTY is where the main marks are: the report and deck must show clearly (a) which prior work and
models exist for plant species identification, (b) which papers and models we built on, (c) what we implemented,
(d) how good ours is against them, and (e) what is new or better in ours. Be convincing AND honest: examiners will
check. Our encoders (BioCLIP 2, DINOv2) and the individual techniques (kNN, linear probe, conformal prediction, geo
priors) are prior work; say so and cite them. The novelty is the system and the evidence, for example: an
uncertainty-aware, evidence-backed pipeline that fuses several scorers on frozen foundation-model features with
calibrated confidence, conformal "not sure" sets, open-set "not in catalog" rejection, hierarchical backoff, a GBIF
location prior, multi-photo fusion and adding a species from a few photos with no retraining; India-focused coverage
(vl600: ~600 species incl. cultivated and common Indian plants, which general benchmarks under-represent); a test split
of observations after 2025-07-01 to reduce overlap with the encoders' training data; and the measured gain over the
strongest open baseline on the SAME test set. Only claim what the code and runs/ support.
Do:
(1) Report: add a section "Related work and what is new" right after the Introduction (renumber), about 2-3 pages:
 - Prior approaches, grouped: hand-crafted leaf features (e.g. LeafSnap); CNNs trained per dataset (Pl@ntNet-300K
   ResNets, Indian medicinal leaf CNNs such as DIMPSAR / MED117 / HerbLife work: usually tens of species, clean leaf
   photos, very high accuracy that does not transfer to field photos); production apps (Pl@ntNet, iNaturalist/Seek,
   Google Lens: closed, single answer, no calibrated uncertainty); foundation models (CLIP, BioCLIP, BioCLIP 2,
   DINOv2/DINOv3, PlantCLEF frozen-backbone + head winners); general VLMs (VLM4Bio: poor at open-ended species ID);
   uncertainty work (split conformal prediction, open-set recognition); geo priors (Mac Aodha et al., SINR).
 - A table "Prior work vs VanaLens": rows = representative systems/papers, columns = species scale, India coverage,
   field photos, calibrated confidence, prediction sets, rejects unknown species, add species without retraining,
   shows evidence, uses location, multi-photo. Fill only what the papers state; "not reported" where unknown.
 - Reported accuracies of prior work ONLY with the dataset they were measured on, in a separate clearly labelled
   table ("reported on their own datasets, not comparable"); never put them in the same column as our numbers.
 - "What we built on" (BioCLIP 2, DINOv2, conformal, GBIF ...) vs "What we implemented" (list our own components and
   scripts) vs "What is new" (4-6 numbered contributions, each tied to a measured number or a working feature).
 - Every citation must be real: use the links in docs/RESEARCH.md; if you have web access, verify title, authors,
   year and venue; if you cannot verify one, drop it. Never invent a paper, author or number.
(2) Head-to-head on OUR vl600 test set: build the comparison table from the baselines rows in runs/L6/final.json
   (L6-d adds them: BioCLIP 2 zero-shot, BioCLIP-only probe, DINOv2-only probe, kNN-only, concat probe, full VanaLens
   1 and 3 photos, with 95% CIs). If they are not there yet, wire the table and a bar chart (with CI error bars) through
   build_report.py placeholders from runs/ so L7-d's rebuild fills them; use runs/L2/b2_vl600.json and
   runs/L3/ablation_vl600.json for what already exists. Also compare capabilities none of the baselines have
   (set size at 90% coverage, ECE, open-set AUROC, few-shot new species).
(3) Deck (scripts/build_deck.py, same clean theme): add 2-3 slides, deck may grow to 15 slides: "Prior work"
   (compact capability matrix), "What is new in VanaLens" (contributions, max 4 bullets), "Head-to-head on the same
   test set" (bar chart with CIs vs BioCLIP 2 zero-shot and single-encoder baselines). Speaker notes explain the
   novelty in plain words. Re-export pitch/preview/ and check the new slides visually.
(4) Rebuild the PDF; tests must pass; no em dashes.
DONE entry Results: list of contributions claimed, number of verified citations, which comparison numbers are filled.
When done, inside `.lanes/L7`: DONE entry in worklog/L7.md, commit with prefix [L7], `git push origin HEAD:lane/L7-pitch`, stop.
```

**L7-d**
```
Lane L7, card L7-d. Setup, from the project root: `& $env:VANALENS_PY scripts/lane.py L7 --start L7-d` (bash: `"$VANALENS_PY" scripts/lane.py L7 --start L7-d`). It prepares your lane folder `.lanes/L7` (git worktree with the latest lane/L7-pitch and main merged in) and commits + pushes your STARTED entry. From then on work ONLY inside `.lanes/L7`: cd into it, read and edit only files under it, run python/pytest there with `$VANALENS_PY`. Never edit files outside `.lanes/L7` (AGENTS.md section 1).
You are lane L7 (pitch). Read AGENTS.md, pitch/report/VanaLens_report.md, the slide PNGs in pitch/preview/ and
runs/L6/final.json. Act as a skeptical reviewer of the report and the deck: write pitch/report/review.md listing
weaknesses (claims without numbers, missing CIs, training-data contamination, dataset and geographic bias, few test
photos per species, baselines not run, unfair comparisons, overclaiming, unclear explanations, slide design issues).
Fix what the existing results in runs/ can fix (wording, missing numbers, CIs, figures, layout); move everything else
into the report's Limitations section. Rewrite pitch/demo.md to cover: a known plant; an unknown plant (rejected,
then added with 5 photos and recognised); a 3-photo plant; a photo with GPS (show the location prior effect); and a
fallback if Wi-Fi fails. First refresh all numbers from the re-run results (L6-d runs/L6/final.json, L3-d
ablations): rebuild the deck charts and slide numbers (scripts/build_deck.py) and the report tables, and update any text whose
claim changed (which metrics reach 95%, open-set, live demo). Rebuild the deck (re-export and re-check slide PNGs) and the PDF. No em dashes.
When done, inside `.lanes/L7`: DONE entry in worklog/L7.md, commit with prefix [L7], `git push origin HEAD:lane/L7-pitch`, stop.
```
