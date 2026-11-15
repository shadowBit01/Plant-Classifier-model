<!--
Source of the VanaLens project report. Build: python pitch/report/build_report.py
Every result number is a {{...}} placeholder filled from runs/*.json at build time (see build_report.py docstring).
Do not type result numbers by hand. No em dashes.
-->

<div class='titleblock'>

# VanaLens

<p class='subtitle'>Calibrated, evidence-backed plant species identification from phone photos: {{row:vl600:VanaLens up to 3 photos + geo:top1}} top-1 on 607 species with up to three photos, from frozen foundation features and light trained heads</p>

<p class='meta'>Project report, November 2026. Code: <a href='https://github.com/shadowBit01/Plant-Classifier-model'>github.com/shadowBit01/Plant-Classifier-model</a>. Results built from <code>runs/</code> (state of {{d:state_created}}).</p>

<div class='abstract'>

**Summary.** VanaLens identifies 607 common Indian and worldwide plant species from one to three ordinary phone photos and says how sure it is. On {{d:n_photos_vl}} test photos taken after its encoders' training data was collected, it reaches {{rowci:vl600:+ geo prior:top1}} top-1 from one photo with location and {{rowci:vl600:VanaLens up to 3 photos + geo:top1}} with up to three photos of the same plant, which meets our 95% target with the whole interval above it (from one photo it is just below the target); top-5 is {{row:vl600:+ geo prior:top5}} from one photo. On the same photos, VanaLens is {{d:h2h_gain_zs}} above BioCLIP 2 zero-shot over the same 607 names (one photo, no location), and {{d:h2h_gain_best3}} above the strongest trained baseline, a linear probe on the same features, when it uses up to three photos and location (from a single photo without location the gain over that probe is {{d:h2h_gain_best}}, with overlapping intervals). Beyond accuracy, VanaLens gives outputs the baselines do not: calibrated confidence (ECE {{pct:L3/ablation_vl600.json:rows.ensemble.ece}}), a genus or family answer when the species is unclear ({{row:vl600:+ geo prior:backoff_acc}} correct at the level it answers), a 90% prediction set, a "not in catalogue" flag (AUROC {{pct:L6/final.json:extras.vl600.open_set.auroc}}), the nearest reference photos and matched traits as evidence, and new species added from five photos with no retraining ({{pct:L6/final.json:extras.vl600.few_shot.2.top1_new}} top-1 on 61 species it had never seen, known-species accuracy kept at {{pct:L6/final.json:extras.vl600.few_shot.2.top1_known_after}}). It is built on frozen BioCLIP 2 and DINOv2 features with trained heads, fusion weights, temperature and thresholds, so it is prepared on a 6 GB laptop GPU and answers on a CPU. Every number carries a 95% bootstrap confidence interval, and Section 8 sets out the current scope with a next step for each item.

</div>
</div>

<!-- pagebreak -->

## Introduction

### The problem

India has more than 18,000 species of flowering plants, and the people who most often need a plant name are not botanists: a farmer checking a weed, a student on a field trip, a buyer of raw medicinal herbs who wants to know that the leaves in the sack are the species on the label, a forest guard logging what grows along a transect. Existing apps answer with a single name and a percentage. When the photo is ambiguous they still give one confident name, they rarely show why, and their confidence numbers are not calibrated: "90%" does not mean right nine times out of ten. Adding a local species usually means waiting for the vendor to retrain.

### Who it is for

- Field users (students, citizen scientists, forest and agriculture staff) who need a name plus a clear signal of when to trust it.
- Quality checkers in the herbal supply chain, where look-alike species are a known adulteration route and an explanation (which traits matched, which reference photos it resembles) matters as much as the label.
- Small institutions (a campus, a botanical garden, a state forest department) that want to add their own species from a handful of photos, without machine-learning staff or GPUs.

### Goals

1. **Accurate:** 95% top-1 on common Indian plants, measured on photos taken after the encoders' training data was collected.
2. **Honest:** calibrated confidence, a prediction set with a coverage guarantee, a genus or family answer when the species is unclear, and an explicit "not in our flora" outcome.
3. **Explainable:** every answer comes with the nearest reference photos and the visual traits that matched.
4. **Extensible:** add a species with a few photos and no retraining.
5. **Cheap:** run on one laptop with a 6 GB GPU for preparation and a CPU for inference.

### Contributions at a glance

Each contribution is backed by a measured number on the vl600 test split ({{d:n_photos_vl}} photos observed on or after 1 July 2025) or by working code; Section 2.5 gives the details.

1. **One calibrated, uncertainty-aware identification system on frozen foundation features.** Probe, MLP, gallery and text-trait scorers are fused and temperature-calibrated, then combined with a 90% conformal set, genus and family backoff, an open-set check, a GBIF location prior and multi-photo fusion. Evidence: ECE {{pct:L3/ablation_vl600.json:rows.ensemble.ece}}; {{rowci:vl600:+ geo prior:backoff_acc}} of answers correct at the rank reported (Tables UNC, CAPS).
2. **A measured head-to-head on the same test photos.** Evidence: {{d:h2h_gain_zs}} over BioCLIP 2 zero-shot with the same 607 names (one photo, no location); {{d:h2h_gain_best3}} over a linear probe on the same features with up to three photos and location (Table H2H).
3. **Multi-photo fusion and location in one calibrated scale.** Evidence: {{rowci:vl600:VanaLens up to 3 photos + geo:top1}} top-1 with up to three photos, the 95% target met with the whole interval above it (Table TARGET).
4. **New species in seconds, without retraining.** Evidence: {{pctci:L6/final.json:extras.vl600.few_shot.2.top1_new}} top-1 on 61 unseen species from five photos each, scored against all 607 classes, with known-species accuracy kept at {{pct:L6/final.json:extras.vl600.few_shot.2.top1_known_after}} (Table FEWSHOT).
5. **Evidence with every answer.** The app shows the nearest gallery photos and the best-matching visual traits (working feature, Figure@app).
6. **vl600, an India-focused benchmark with a date cut.** 607 species chosen only by observation count, covering {{num:L1/vl600_stats.json:coverage_pct:.1f}}% of India's research-grade plant observations on iNaturalist, with every test photo taken after the encoders' training data (Table DATASET).

This report places VanaLens against prior work and states what is new (Section 2), describes how it works (Sections 3 and 4), the data (Section 5), how we evaluated it (Section 6), what we measured (Section 7), the current scope (Section 8) and the next steps (Section 9).

## Related work and what is new

VanaLens is a systems contribution. Its encoders and each of its individual techniques come from earlier work, which we cite below. What is new is how they are put together into one system with measured uncertainty, and the evidence we provide for it on an India-focused test set built to limit overlap with the encoders' training data. This section separates the three: what exists, what we built on, and what we added. Codes in square brackets (R1, R2, ...) point to the related-work references at the end of the report.

### Prior approaches to plant species identification

**Hand-crafted leaf features.** Leafsnap [R1] was the first mobile app to identify plants by automatic visual recognition: it rejects non-leaf images, segments a leaf photographed on a plain background and matches multi-scale contour curvature, covering 184 tree species of the north-eastern United States. Systems of this kind need a single leaf on an untextured background, which is not how people photograph plants in the field.

**CNNs trained per dataset.** The dominant approach since then is to train or fine-tune a convolutional network (or, more recently, a vision transformer) on one dataset. Pl@ntNet-300K [R2] is the most careful public benchmark of this kind: 306,146 citizen-science photos of 1,081 species with a heavy long tail and many look-alikes, evaluated with set-valued metrics because many photos are ambiguous even for experts. For Indian plants, most published work trains a CNN on a few dozen medicinal species: for example a MobileNet reaching 98.3% on six species with 500 images each [R3]. The public Indian datasets are of the same scale: DIMPSAR [R4] (40 species as whole plants, 80 as single leaves, smartphone photos with varied backgrounds), MED117 [R5] (about 117 species, frames cut from leaf videos on a white background) and HerbLife [R6] (112 species from Kottayam, Kerala). These papers report very high accuracy on clean, small, closed sets; none of them measures transfer to ordinary field photos, and adding a species means retraining.

**Production apps.** Pl@ntNet [R7], iNaturalist's Seek and Google Lens identify plants at large scale from field photos. Their current models and evaluations are not public (the Pl@ntNet paper describes its collaborative identification workflow), and we found no published description of a coverage guarantee, an explicit "not in our list" outcome or a way for a user to add a local species in any of them.

**Foundation models.** CLIP [R11] showed that contrastive image-text training gives features that transfer zero-shot. BioCLIP [R10] and BioCLIP 2 [R9] apply this to the tree of life: BioCLIP 2 is trained on 214 million organism images with their taxonomic names and reports an 18.0-point gain over BioCLIP on zero-shot species classification. Self-supervised DINOv2 and DINOv3 [R12] give strong frozen visual features; the DINOv3 authors recommend frozen features with simple heads. The strongest PlantCLEF teams do exactly this: the second-placed 2025 team used the organisers' DINOv2 backbone with light heads and image tiling [R13]. These are encoders or competition pipelines; they do not output calibrated confidence, prediction sets or evidence.

**General vision-language models.** VLM4Bio [R14] evaluated twelve VLMs on biological images (fish, birds and butterflies, no plants). GPT-4V names a bird species correctly 17.4% of the time when asked open-ended, but 82.6% of the time when it chooses from a list, while BioCLIP reaches 67.1% open-ended. VLMs are poor at naming species but useful as choosers over a short list, which is why VanaLens only allows a VLM to rerank a shortlist (that optional step was not evaluated here).

**Uncertainty.** Split conformal prediction [R15] turns any classifier into one that outputs a set guaranteed to contain the true class with a chosen probability, assuming calibration and test data are exchangeable; adaptive variants exist [R15]. For plant identification, recent work adapts conformal sets to long-tailed classes [R16] and aggregates several photos of one plant [R17], both evaluated on Pl@ntNet-300K. Open-set recognition [R20] asks a model to say "none of the known classes"; a simple confidence or similarity threshold is the standard baseline. Temperature scaling [R21] is the standard fix for over-confident networks.

**Location priors.** Mac Aodha et al. [R18] showed that multiplying image probabilities by a learned "is this species found here?" prior gives large gains on fine-grained datasets; SINR [R19] estimates ranges for tens of thousands of species from presence-only data. Multi-organ studies show that fusing several photos of one plant helps strongly: top-1 rises from 77% for the whole plant to 97% when all five fixed views are fused [R8].

{{table:reported}}

Table REPORTED lists headline accuracies of prior work. They were measured on different datasets, with different numbers of species, image types and protocols, so they cannot be ranked against each other or against our results. The comparison that can be made fairly is on one test set, in Section 2.4.

{{table:prior}}

Table PRIOR summarises these works. Each column is filled only from what the paper or product page states. Among the works we reviewed, none reports all of the following in one system: calibrated confidence, a prediction set, an unknown-species outcome, adding species without retraining, visible evidence, a location prior and multi-photo fusion, on field photos of Indian plants.

### What we built on

| Component | Prior work we use | Status in VanaLens |
|---|---|---|
| Biology-aware image and text encoder | BioCLIP 2 ViT-L/14 [R9] | used frozen, weights unchanged |
| Second, purely visual encoder | DINOv2-large [R12] (DINOv3 planned; gated weights) | used frozen, weights unchanged |
| Nearest-neighbour classification and linear probe on frozen features | standard practice; PlantCLEF frozen backbone + head [R13] | re-implemented on our features |
| Scoring by text descriptions of traits | classification by description [R22] | re-implemented with BioCLIP 2's text tower |
| Calibration | temperature scaling [R21] | re-implemented |
| Prediction sets | split conformal prediction [R15] | re-implemented |
| Open-set threshold | max-similarity baseline [R20] | re-implemented |
| Location prior | geo priors [R18]; occurrence counts from GBIF | re-implemented with GBIF counts |
| Multi-photo fusion | multi-organ evidence [R8] | re-implemented (mean of log-probabilities) |
| Photos and metadata | iNaturalist [R23], iNaturalist Open Data, GBIF | used as data sources |

### What we implemented

All of the following was written for this project (code in the repository; paths relative to its root):

1. **Dataset builder** `eval/build_vl600.py`: selects species by observation count only, downloads licensed photos, and splits by observation and by date (test photos observed on or after 1 July 2025). Also `scripts/make_manifests.py` for inat_india.
2. **Feature extraction** `vanalens/encoders.py`, `scripts/extract_all.py`: fp16 BioCLIP 2 + DINOv2 embeddings with flip test-time augmentation and augmented gallery views, guarded by weight-integrity tests (Section 4.13).
3. **Scorers** `vanalens/scorers.py`: linear probe, MLP head, kNN gallery, text-trait prototypes; trait lists and prompts in `vanalens/traits.py`, `scripts/build_traits.py`.
4. **Fusion, calibration and sets** `vanalens/fusion.py`, `vanalens/conformal.py`, `scripts/fit_core.py`: log-linear fusion with weights searched on the calibration split, temperature, split conformal threshold, open-set threshold.
5. **Hierarchy and location** `vanalens/hierarchy.py` (species, genus, family backoff), `vanalens/geo.py` (GBIF prior with a strength fitted on calibration).
6. **End-to-end pipeline** `vanalens/pipeline.py`: one `VanaLens` object used by both the evaluation and the app, with multi-photo fusion and `add_species` (append photos, no retraining).
7. **Evaluation** `eval/metrics.py`, `eval/harness.py`, `scripts/final_table.py`: every metric with a 95% bootstrap interval over observations, plus open-set and few-shot experiments.
8. **App** `app/app.py`: photo in; species or genus, confidence, set, nearest reference photos and matched traits out.

### How good it is against them, on the same test set

We cannot run the closed apps on our photos (the Pl@ntNet API needs a key that was not available), and we do not quote other papers' accuracies as if they were comparable. Instead every baseline below is run on **the same {{d:n_photos_vl}} vl600 test photos**, with the same frozen embeddings and the same gallery and calibration splits as VanaLens, and nothing is tuned on test.

{{table:h2h}}

{{fig:h2h}}

With the same input as the baselines (one photo, no location), VanaLens is {{d:h2h_gain_zs}} above BioCLIP 2 zero-shot over the same 607 names, the strongest open baseline that needs no training; with the location prior the gap is {{d:h2h_gain_zs_geo}}, and with up to three photos {{d:h2h_gain_zs3}}. Against the strongest baseline in Table H2H ({{d:h2h_best}}, {{d:h2h_best_top1}}), the single-photo gain without location is {{d:h2h_gain_best}}; {{d:h2h_sep}}. Location raises it to {{d:h2h_gain_best_geo}} and up to three photos to {{d:h2h_gain_best3}}. All rows share the same bootstrap resamples, but we did not compute an interval for the paired difference, so overlap does not prove the methods are equal. The gain over BioCLIP 2 zero-shot shows what a gallery of reference photos and trained heads add to the same encoder; the concatenated-feature probe is the stricter comparison, since it shares our features and training photos. This is the expected profile of a frozen-feature design: strong features with a trained head carry most of the single-photo accuracy, fusion and calibration add a little top-1 and make the confidence usable, and the two kinds of extra context VanaLens can use, location and more photos of the same plant, add the most. {{d:h2h_pending}}

The larger difference is in what the baselines cannot output at all (Table CAPS): a 90% set, an answer at genus or family level when the species is unclear, an unknown-species flag, a new species from a few photos without retraining, multi-photo fusion and evidence for the answer. On calibration alone all methods in Table CAPS do well on this test set, with VanaLens the lowest; the difference is that VanaLens turns calibrated scores into decisions (sets, backoff, the unknown flag).

{{table:caps}}

### What is new in VanaLens

We claim the following contributions. Each is tied to a number in this report or to a working feature in the code.

1. **One uncertainty-aware system on frozen foundation features with light trained heads.** Scorers on frozen BioCLIP 2 + DINOv2 features are fused with weights chosen on the calibration split (the final model uses {{d:ens_weights}}; the kNN gallery, weighted zero there, supplies the evidence photos, the open-set score and new species) and temperature-calibrated, then followed by a 90% conformal set, a genus or family backoff, an open-set check, a GBIF location prior and multi-photo fusion. Calibration error of the final model is {{pct:L3/ablation_vl600.json:rows.ensemble.ece}} ECE on vl600 test, and answers are correct at the rank VanaLens chooses to report {{row:vl600:+ geo prior:backoff_acc}} of the time. The individual parts are prior work (Section 2.2); the combination and its joint evaluation are ours.
2. **Adding a species in seconds, with no retraining, measured.** Appending five photos of a species it has never seen gives {{pct:L6/final.json:extras.vl600.few_shot.2.top1_new}} top-1 on 61 held-out species, scored against all 607 classes, while accuracy on known species stays at {{pct:L6/final.json:extras.vl600.few_shot.2.top1_known_after}}. Per-dataset CNNs (Table PRIOR) must be retrained for this.
3. **Evidence with every answer.** The app shows the nearest reference photos from the gallery and the visual traits that matched best, so a user can check the answer. This is a working feature, not a measured number.
4. **Multi-photo fusion and a location prior in the same calibrated scale.** Up to three photos of one plant raise top-1 to {{rowci:vl600:VanaLens up to 3 photos + geo:top1}}, and the GBIF prior {{d:geo_effect_vl}} on vl600.
5. **vl600, an India-focused test with a date cut.** 607 species chosen only by observation count ({{num:L1/vl600_stats.json:coverage_pct:.1f}}% of India's research-grade plant observations on iNaturalist), including cultivated and common Indian plants that general benchmarks under-represent, with every test photo observed on or after 1 July 2025 to reduce overlap with the encoders' training data, and all results reported with 95% confidence intervals over observations.
6. **A measured head-to-head on that test set.** From one photo without location VanaLens is {{d:h2h_gain_zs}} above BioCLIP 2 zero-shot and {{d:h2h_gain_best}} above the best single baseline; with up to three photos and location, {{d:h2h_gain_zs3}} and {{d:h2h_gain_best3}} (Table H2H).

Scope of these claims: we do not claim a new architecture, a better encoder or a state-of-the-art result on a public benchmark. The open-set check is a baseline-strength detector (AUROC {{pct:L6/final.json:extras.vl600.open_set.auroc}}, Section 7.6), the 90% sets are almost always a single species because accuracy already exceeds 90% (Section 7.5), and no production app was compared on the same photos (Section 8).

## How the system works

This section follows one prediction from photo to answer, without equations. Figure@pipeline shows the same steps.

{{fig:pipeline}}

**1. The user takes photos.** Say a student photographs a flowering shrub by a road in Bhopal: one photo of the flower cluster and one of a leaf. The phone stores the GPS position in the photo.

**2. Is it a plant at all?** The first check compares the photo with short text descriptions ("a photo of a leaf", "a photo of a flower", ..., "not a plant") using BioCLIP 2, a model trained on images and names from across the tree of life. This also tells the app which organ it is looking at.

**3. Turn each photo into numbers.** Two frozen encoders read each photo. BioCLIP 2 produces a 768-number summary that captures what kind of organism it is; DINO, a general self-supervised vision model, produces a 1024-number summary of fine visual detail (texture, venation, margins). Both encoders stay frozen, so these summaries are computed once and reused by every later step. The two summaries are joined into one 1792-number vector per photo.

**4. Ask several experts.** Four scorers look at that vector and each give a probability for every species in the catalogue:

- a **linear probe** and a small **MLP**, light classifiers trained on top of the frozen vectors;
- a **gallery search** (k nearest neighbours), which finds the most similar reference photos of known species;
- **text-trait prototypes**: for each species we wrote short visual descriptions ("leaves opposite, with a toothed margin", "flowers in dense orange-red clusters"), encoded them with BioCLIP 2's text side, and score how well the photo matches each species' description.

**5. Combine and calibrate.** The scorers' opinions are combined by a weighted vote on the log scale, with weights chosen on a held-out calibration split. A single temperature number then rescales the result so that a stated 80% is right about 80% of the time.

**6. Use the location.** From the photo's GPS, VanaLens looks up how often each species has been recorded nearby in GBIF, the global biodiversity database, and gently boosts plants that are common there and lowers plants that are not. A look-alike from another continent drops down the list.

**7. Combine photos of the same plant.** The flower photo and the leaf photo give two probability lists; VanaLens averages them on the log scale, so a species must look right in both photos to stay on top.

**8. Decide what to say.** From the final probabilities VanaLens builds a **90% prediction set** (a list of species that contains the true one at least 90% of the time), checks whether the photo is close enough to anything in the gallery to be a known species at all (**open-set check**), and picks the deepest rank it is confident about: the species if one species has at least 60% of the probability, otherwise the genus, otherwise the family.

**9. Show the evidence.** The answer is shown with its calibrated confidence, the prediction set, the three nearest reference photos from the gallery and the traits that matched best (Figure@app). If the set holds more than one species, an optional vision-language model can choose among the short list; that step needs an API key and was not used in the results below.

{{fig:app}}

## Approach and methods

### Design choice: frozen foundation features with light trained heads

VanaLens keeps its two foundation encoders frozen and trains light components on their features. This is a deliberate design, chosen for five practical benefits:

- **Real training where it pays off.** We train a linear probe and a one-hidden-layer MLP on {{d:train_rows_vl}} feature vectors ({{d:gallery_vl}} gallery photos plus two augmented views of each), and fit log-linear fusion weights, a temperature, a conformal threshold, a location-prior strength and an open-set threshold. All of these are chosen on the gallery and calibration splits, never on test.
- **New species in seconds.** A species is added by appending its feature vectors to the gallery and a text prototype; no gradient step is taken, so the cost is encoding the new photos (the live pipeline encodes and answers an observation of up to three photos in {{d:e2e_sec_vl}} on a laptop CPU). Section 7.7 measures the accuracy this gives.
- **Runs on modest hardware.** Features are extracted once on a 6 GB laptop GPU; the trained heads are saved as plain arrays, so inference needs only numpy and runs on a CPU.
- **Explanations that are real.** The evidence photos are actual gallery neighbours in the same feature space, and the matched traits are the text prompts that scored highest, so what the user sees is what the system compared.
- **Consistent with current practice.** The DINOv3 authors recommend frozen features with simple heads [R12], and strong PlantCLEF systems use frozen DINOv2 backbones with light heads [R13].

The trade-off is that the encoders' features are not specialised to these 607 species. Table H2H shows how far frozen features already go (a linear probe on them reaches {{d:h2h_best_top1}}); a fine-tuned encoder of the same size is a useful reference point that we list as a next step (Sections 8 and 9).

### Frozen encoders

**BioCLIP 2** (`hf-hub:imageomics/bioclip-2`, ViT-L/14, 768-d) is a CLIP-style model trained on TreeOfLife-200M, 214 million organism images with their full taxonomic names. It has both an image and a text tower, which is what makes zero-shot naming and text-trait prototypes possible. **DINO** supplies the second, purely visual view. The design uses DINOv3 ViT-L/16 (1024-d), but its weights are gated on Hugging Face and no licence acceptance was available during the build, so the documented fallback `facebook/dinov2-large` (also 1024-d) is used. Images are encoded in fp16 with horizontal-flip test-time augmentation; for the gallery we also store two augmented views per photo to give the heads more training rows. Each encoder's block is L2-normalised on its own before the two are concatenated.

### Linear probe and MLP

The linear probe is a multinomial logistic regression on the 1792-d vector, trained in PyTorch with AdamW on the gallery photos (plus their augmented views) and early-stopped on the calibration split. The MLP adds one hidden layer of 1024 units with ReLU and dropout 0.5. For both, weight decay, label smoothing and class-balanced loss weights are chosen by a small grid search on calibration top-1. The trained heads are saved as plain arrays, so inference needs only numpy.

### Nearest-neighbour gallery

The gallery holds every reference photo's vector. For a query, the top k = 10 neighbours by cosine similarity each add exp(similarity / 0.05) to their species' score, which is then normalised. The gallery has three jobs even when its fusion weight is small: it supplies the evidence photos, its best similarity is the open-set score, and it is where new species are added.

### Text-trait prototypes

Following "classification by description" [R22], each species gets several prompts of the form "a photo of *Ocimum tenuiflorum*, which has *trait*" plus a plain "a photo of *Ocimum tenuiflorum*". The prompts are encoded with BioCLIP 2's text tower and a photo is scored by its similarity to each species' prompts (temperature 0.01). The trait descriptions were generated with a language model and are pending botanist review; species without generated traits use a taxonomic prompt ("a photo of family genus species, common name"). This scorer needs no training images at all, and its best-matching prompts are the "matched traits" shown to the user.

### Log-linear fusion and temperature scaling

Scorer outputs are combined by log-linear pooling, P = softmax(sum of w_i log P_i), with weights searched on a simplex grid (step 0.1) to maximise calibration-split top-1, ties broken by lower negative log-likelihood. A weight of zero removes a scorer. On vl600 the calibration split chose the "{{num:L3/ablation_vl600.json:final_row:s}}" configuration; the four-scorer ensemble uses weights {{d:ens_weights}} and the three-scorer fusion {{d:fused_weights}}. A single temperature T, chosen on the calibration split by negative log-likelihood ({{num:L3/ablation_vl600.json:state.ensemble.T:.3f}} on vl600), then rescales the fused probabilities. Temperature scaling does not change which species ranks first; it only makes the confidence honest.

### Split conformal prediction sets

For a 90% guarantee (alpha = 0.1) we use split conformal prediction: on the calibration split compute the score 1 minus the probability of the true species, and take its ceil((n+1)(1-alpha))/n quantile, q-hat ({{num:L3/ablation_vl600.json:state.ensemble.qhat:.3f}} on vl600). At test time the set holds every species with probability at least 1 minus q-hat, and always the top species. If calibration and test photos are exchangeable, the set contains the true species at least 90% of the time.

### Open-set reject

A photo whose best gallery cosine similarity falls below a threshold is flagged as possibly "not in our catalogue". The threshold is the 5% quantile of the best similarity of known-species calibration photos, so 95% of known calibration photos pass. Section 7.6 measures it; it is the component with the most room to improve.

### Hierarchical backoff

Species probabilities are summed into genus and family probabilities using the taxonomy in `labels.json`. VanaLens reports the deepest rank whose best probability is at least 0.6: species, else genus, else family, else "unknown". A correct genus is useful (many medicinal look-alikes are congeners), and it is better than a wrong species.

### GBIF location prior

For a photo with GPS, GBIF's occurrence search returns the number of plant records per species within a radius of the point; results are cached per 0.5 degree cell. The prior for species c is its count plus a smoothing constant, normalised; species without a GBIF key get only the smoothing. The final probability is P times prior to the power beta, renormalised. Beta is picked from a small grid by negative log-likelihood on the calibration split: {{d:beta_vl}} on vl600 and {{d:beta_in}} on inat_india. Beta = 0 would switch the prior off.

### Multi-photo fusion

For up to three photos of one plant, VanaLens averages the calibrated per-photo log-probabilities, renormalises, and applies the location prior once. Averaging (rather than summing) keeps the confidence on the same calibrated scale whatever the number of photos.

### Adding species without retraining

`add_species` embeds the new photos, appends them to the kNN gallery and adds a "a photo of <species>" text prototype. The heads were trained before the species existed, so for new classes they defer to the gallery: the probability mass the gallery gives to new species is passed through and the heads' own probabilities are scaled down to make room. Nothing is retrained.

### Engineering safeguard: encoder weight integrity

Frozen features are only as good as the weights they are computed with, so the encoder loader is guarded by tests (`tests/test_l2_weights.py`): the loader raises an error if any parameter is left unloaded, the DINOv2-large checkpoint must load with zero unloaded tensors, and two independent loads of the encoders must give identical embeddings. Every cached embedding and every number in this report was computed with these checks in place, and the live end-to-end run (Section 7.9) confirms that the app's live features agree with the cached ones.

## Dataset

### vl600: the main dataset

vl600 was built for this project automatically from iNaturalist (Table DATASET). Species are chosen **by observation count only**, with no hand-picking: (a) the 400 most-observed research-grade plant species in India, (b) the 200 most-observed plant species worldwide that are not already in (a), and (c) a fixed list of common Indian plants (tulsi, neem, hibiscus, mango, peepal, banyan, curry leaf and others) resolved to accepted iNaturalist names, of which 7 were not already covered. The resulting {{num:L1/vl600_stats.json:species:,d}} species account for {{num:L1/vl600_stats.json:coverage_pct:.1f}}% of all research-grade plant observations in India.

For each species we took up to 80 photos observed **before 1 July 2025** for the gallery and calibration splits, and up to 15 observations made **on or after 1 July 2025** (up to 3 photos each) for the test split, preferring Indian observations and filling from worldwide ones. BioCLIP 2's training data was collected before that date, so the test photos cannot be in its training set; this matters because BioCLIP 2 is trained on GBIF and iNaturalist images, and an older test photo might have been seen during pre-training. Only photos licensed CC0, CC BY or CC BY-NC are used. Pre-cutoff observations are split 80/20 into gallery and calibration **by observation**, so two photos of the same plant never end up on both sides; the test split is defined by date alone. Every species has at least 5 test observations ({{num:L1/vl600_stats.json:species_lt5_test_obs:d}} species fall short).

{{table:dataset}}

{{fig:photos}}

### inat_india: the regional check

inat_india was the project's first dataset: the 60 most-observed research-grade plant species in Madhya Pradesh, with GPS on every photo, up to 30 observations before 1 June 2025 (gallery and calibration) and up to 15 observations from 1 July 2025 (test), one photo per observation. Species with fewer than 5 test observations were dropped, leaving {{d:classes_in}}. It is small, so its confidence intervals are wide, but it is entirely local and every photo has coordinates, which makes it the natural check of the location prior.

### Access and rebuilding

Photos and metadata come from [iNaturalist](https://www.inaturalist.org) through its public API v1 (`api.inaturalist.org`, at most one request per second) and the photo files from the [iNaturalist Open Data bucket on AWS](https://registry.opendata.aws/inaturalist-open-data) (`inaturalist-open-data.s3.amazonaws.com`). We do not redistribute the images; each one keeps its observer's licence, and `meta.csv` records observation id, date, coordinates, licence and source URL for every photo. To rebuild everything from scratch:

1. `python -m eval.build_vl600` (species selection, download, manifests, `runs/L1/vl600_stats.json`); `python scripts/make_manifests.py` for inat_india.
2. `python scripts/extract_all.py` (frozen embeddings for every split, GPU).
3. `python scripts/build_traits.py --dataset vl600` (text prototypes) and `python scripts/fit_core.py --dataset vl600` (heads, fusion, temperature, conformal, open-set threshold).
4. `python -m vanalens.pipeline fit-beta --dataset vl600` (location prior strength), then `python scripts/final_table.py` (all tables with CIs, `runs/L6/final.json`).

## Experimental setup

### Protocol

All choices (head hyper-parameters, which head is the default, fusion weights, temperature, conformal q-hat, open-set threshold, location beta, selective threshold) are made on the calibration split. The test split is only evaluated, once per configuration. The numbers in Section 7 are computed from the cached frozen embeddings through the same code the live app runs (`VanaLens.scores`, then `VanaLens.combine`, then the conformal set and backoff), so the table and the app agree by construction.

### Metrics

- **Top-1 / top-5 accuracy:** the true species is the first / among the first five.
- **Macro top-1:** per-species top-1 averaged over species, so common species do not dominate.
- **Genus / family top-1:** accuracy after summing species probabilities to genus or family.
- **Coverage and mean size of the 90% conformal set.**
- **Backoff accuracy:** share of answers that are correct at the rank VanaLens chose to report; **species share:** how often that rank is the species.
- **ECE:** expected calibration error, the average gap between stated confidence and actual accuracy over confidence bins; **NLL:** negative log-likelihood.
- **Selective accuracy:** accuracy over the photos whose confidence clears a threshold chosen on calibration to reach 95%, and the share of photos answered.
- **Open-set:** reject rate on species removed from the model, false-reject rate on known species, and AUROC of the similarity score.
- **Few-shot:** top-1 on species added after training with 1, 3, 5 or 10 photos each.
- **Latency:** seconds per observation in the live pipeline.

### Confidence intervals

Every VanaLens number carries a 95% percentile bootstrap interval from 1,000 resamples of test **observations** (not photos, since photos of one plant are correlated), seed 0. One resample matrix is shared by all metrics of a dataset. The baselines run uses the same resamples, so its rows carry intervals too; only BioCLIP 2 zero-shot, whose result file has no per-photo predictions, is a point value. We report each method's own interval, not an interval for the difference between two methods, so overlapping intervals are not a test of equality.

### Baselines

- **B0 Pl@ntNet API**, the production identification service. Not run: no API key was available.
- **B2 BioCLIP 2 zero-shot**, the same encoder with no gallery and no training: (i) closed set, choosing among the dataset's species names, and (ii) open-ended over BioCLIP 2's whole tree of life. (In `runs/L6/final.json` this row is labelled "B2 BioCLIP 2 kNN"; the script that produced it, `scripts/baseline_bioclip.py`, runs zero-shot classification.)
- **B3 linear probe**: the linear head alone on our embeddings, the standard frozen-feature baseline, run on BioCLIP 2 features alone, DINOv2 features alone and both concatenated (Table H2H).
- **Single scorers**: kNN gallery alone and text prototypes alone (0 images per species).
- **B4 general vision-language model**, open-ended. Not run: no LLM key.

B1 (Pl@ntNet-300K ResNet weights) was omitted to stay within compute budget (a fine-tuned encoder reference row is listed as a next step in Section 8), and no campus photo set was collected during this build.

## Results

### Main results

{{table:main_vl600}}

{{fig:accuracy}}

On vl600, the single-photo system with location reaches {{rowci:vl600:+ geo prior:top1}} top-1 and {{rowci:vl600:+ geo prior:top5}} top-5. This is {{d:gain_b2_vl}} than BioCLIP 2 zero-shot over the same 607 names; relative to BioCLIP 2's open-ended guess over its whole tree of life ({{pct:L2/b2_vl600.json:tol_top1}}) the gain is much larger, which is the gap between "name this organism" and "choose from a curated regional flora with a gallery". Family-level accuracy is {{rowci:vl600:+ geo prior:family_top1}}.

{{table:main_inat}}

On inat_india the single-photo system reaches {{rowci:inat_india:VanaLens 1 photo (no geo):top1}} without location and {{rowci:inat_india:+ geo prior:top1}} with it. The intervals are wide (only {{d:n_obs_in}} test observations).

### Ablation

{{table:ablation_vl600}}

Table ABLVL shows what each part contributes on vl600. In the current numbers the trained heads are the strongest single scorers, and the text prototypes, which use no training images, reach {{pct:L3/ablation_vl600.json:rows.text.top1}} on their own. Fusion adds a little top-1 over the best single scorer, and {{d:temp_note}}. {{d:knn_note}}

{{table:ablation_inat}}

On the small inat_india set the text prototypes are, in the current numbers, stronger relative to the trained heads ({{pct:L3/ablation_inat_india.json:rows.text.top1}} against {{pct:L3/ablation_inat_india.json:rows.probe.top1}} for the probe), which is what one expects when the gallery has only about a dozen photos per species: the heads have little to learn from, while the descriptions carry prior knowledge. On that set the text scorer alone is as accurate as the full system ({{row:inat_india:VanaLens 1 photo (no geo):top1}}) and BioCLIP 2 zero-shot alone reaches {{row:inat_india:B2 BioCLIP 2 kNN:top1}}, so for 52 common local species most of the accuracy comes from the frozen encoder's prior knowledge, not from anything we fitted.

### One photo against three

Combining up to three photos of the same plant gives the largest single gain in the current numbers. On vl600, going from one photo with location to up to three photos with location {{d:multi_effect_vl}}, to {{rowci:vl600:VanaLens up to 3 photos + geo:top1}} (per observation; {{d:multi_obs_vl}} test observations have 2 or 3 photos, the rest one). This matches the botanical literature, where fusing views of different organs raises accuracy sharply. In inat_india every test observation has exactly one photo, so the multi-photo row equals the single-photo row there.

### Location prior

Adding the GBIF prior {{d:geo_effect_vl}} on vl600 (beta {{d:beta_vl}}) and {{d:geo_effect_in}} on inat_india (beta {{d:beta_in}}). The prior can only help where look-alikes differ in range: vl600 mixes Indian and worldwide species and its test photos come from several countries, while inat_india is a single Indian state where most candidates are recorded nearby, so its prior carries little information and its effect is within the wide confidence interval. On both datasets the intervals with and without the prior overlap: the location prior is a small effect in these numbers, not a headline.

### Uncertainty: sets, backoff, selective answers and calibration

{{table:uncertainty}}

{{fig:calibration}}

Three results stand out. First, calibration is good: after temperature scaling the final ensemble's ECE on vl600 is {{pct:L3/ablation_vl600.json:rows.ensemble.ece}}, against {{pct:L3/ablation_vl600.json:rows.probe.ece}} for the linear probe alone (Figure@calibration). Second, the 90% conformal sets cover the true species {{row:vl600:+ geo prior:coverage}} of the time on vl600 and {{row:inat_india:+ geo prior:coverage}} on inat_india, {{d:sets_note}}. Third, the selective threshold chosen on calibration to give 95% accuracy reaches {{pctci:L6/final.json:extras.vl600.selective.test_accuracy_answered}} on vl600 test, {{d:selective_note}}. Hierarchical backoff, in contrast, works as intended: answers are correct at the reported rank {{rowci:vl600:+ geo prior:backoff_acc}} of the time on vl600, at the cost of answering at genus or family level for the rest.

### Open-set reject

{{table:openset}}

The open-set gate is the component with the most headroom. On vl600, with 61 species removed from the model, it rejects {{pctci:L6/final.json:extras.vl600.open_set.reject_rate_held_out}} of photos of those species while also wrongly rejecting {{pctci:L6/final.json:extras.vl600.open_set.false_reject_rate_known}} of photos of known species (AUROC {{pct:L6/final.json:extras.vl600.open_set.auroc}}). A threshold on one cosine similarity cannot separate a new species from an unusual photo of a known one, and the threshold, set so that 95% of calibration photos pass, does not transfer to the harder test photos. {{d:unknown_case}}

### Adding new species

{{table:fewshot}}

{{fig:fewshot}}

Adding species is where the frozen-feature design pays off. On vl600, the 61 held-out species reach {{pctci:L6/final.json:extras.vl600.few_shot.0.top1_new}} top-1 from one photo each, {{pctci:L6/final.json:extras.vl600.few_shot.2.top1_new}} from five and {{pctci:L6/final.json:extras.vl600.few_shot.3.top1_new}} from ten, measured over all 607 classes, and accuracy on the known species stays at {{pct:L6/final.json:extras.vl600.few_shot.3.top1_known_after}} afterwards. No weights change; the operation is an append to the gallery. New species still trail the trained ones, because for them only the gallery and one name prompt vote, not the trained heads or a full trait list. Two notes on this test: the name prompt of an added species is approximated by the mean of its cached trait prototypes (to avoid loading the text tower on CPU), which carries slightly more information than a name-only prompt; and on inat_india only 5 species are held out (53 test photos), so its 100% five-shot result is indicative only.

### Against the 95% target

{{table:target}}

On vl600 the metrics that reach 95% are: {{d:reach_yes_vl}}. Those that do not are: {{d:reach_no_vl}}. On inat_india the point values that reach 95% are: {{d:reach_yes_in}} (not reaching: {{d:reach_no_in}}), though with only {{d:n_obs_in}} observations the lower end of the interval falls below 95% for {{d:ci_below_in}} metrics.

### Live pipeline and latency

A separate end-to-end run pushes {{d:e2e_n_vl}} vl600 test observations one by one through the live app code (photo files in, answer out), on {{d:e2e_device_vl}}. It takes {{d:e2e_sec_vl}} per observation once the models are loaded. Top-1 in that run is {{d:e2e_top1_vl}} ({{d:e2e_ci_vl}}, Wilson interval; only {{d:e2e_n_vl}} observations, so it shows that the live path agrees with the cached evaluation, not a second accuracy estimate), but {{d:e2e_note}}

### Error analysis

A per-species or per-growth-form breakdown (trees against herbs, grasses, ferns) and a confusion analysis of the hardest look-alike pairs were not measured: the result files store aggregate metrics and the per-photo predictions needed for them are not kept in `runs/`. The aggregates give one hint: family top-1 is {{row:vl600:+ geo prior:family_top1}} against {{row:vl600:+ geo prior:top1}} at species level on vl600, so {{d:fam_recover_vl}} of the species errors still land in the right family. The rest are confusions across families, which a per-species analysis should explain first.

## Limitations: current scope and next steps

The table states the current scope of the system and of its evaluation, one item per row, each paired with a concrete next step. It was written after a skeptical review of the report and the deck (`pitch/report/review.md`) and covers every material point that review raised.

| Current scope | Next step |
|---|---|
| **Test photos come from the same source as the gallery.** All photos are iNaturalist research-grade observations, and the date cut does not exclude the same plant photographed again: {{d:same_spot_vl}} test observations with GPS share rounded coordinates (about 11 m) with a gallery or calibration observation of the same species. | Collect a campus and market test set with ordinary phones, including raw-herb photos, and report it separately; drop test observations within 50 m of a gallery observation of the same species. |
| **The open-set check is baseline strength.** It flags {{pct:L6/final.json:extras.vl600.open_set.reject_rate_held_out}} of photos of 61 unseen species at a {{pct:L6/final.json:extras.vl600.open_set.false_reject_rate_known}} false-flag rate on known species (AUROC {{pct:L6/final.json:extras.vl600.open_set.auroc}}), and close relatives of known species are the hardest case (Section 7.6). | Train a small detector on species held out from the gallery that combines gallery similarity, the energy of the fused logits and the text-prototype margin. |
| **Calibration-to-test shift and set size.** Thresholds chosen on pre-cutoff calibration photos are too lenient for the harder post-cutoff test photos (the selective step answers {{pct:L6/final.json:extras.vl600.selective.test_share_answered}} of them), and with q-hat {{d:qhat_vl}} the 90% sets are single species, valid because accuracy exceeds 90% but adding little information. | Calibrate on a recent post-cutoff slice that is not in test, and add adaptive sets (APS/RAPS) at 95% and 98% coverage, with the long-tail and multi-photo conformal variants of [R16, R17]. |
| **Catalogue size and on-device use.** The catalogue is 607 species, inference takes {{d:e2e_sec_vl}} per observation on a laptop CPU, and the second encoder adds {{d:dino_gain_short}} top-1 over BioCLIP 2 alone while a second ViT-L encoder runs on every photo. | Grow to the 2,000 most-observed Indian species (only gallery photos and traits are needed), test DINOv3 or drop the second encoder, and distil to a small on-device encoder for an offline phone app. |
| **Evaluation breadth.** The Pl@ntNet API and general VLM baselines were not run (no keys), per-photo predictions are not stored, so there is no paired test against the linear probe and no per-species error analysis, and each species has {{d:test_obs_sp_vl}} test observations; {{d:india_test_vl}} of test observations are from India. | Store per-photo predictions to report paired bootstrap intervals, per-species and India-only results; run the Pl@ntNet and VLM baselines and add a fine-tuned-encoder reference row. |
| **Explanations are not yet expert- or user-validated.** Trait descriptions were generated with a language model and are pending botanist review, and the evidence photos come from the kNN gallery, which has weight 0 in the final fusion ({{d:ens_weights}}), so they show similar examples rather than the deciding vote. | Botanist review of the traits of the 100 most-confused species first; show neighbours of the deciding scorer, and run a small user study on whether the evidence helps people catch wrong answers. |

## How we plan to overcome them and improve

The right-hand column of the table above is the plan for each item. Beyond it, three features follow naturally from the design:

- **Offline mobile app.** A small distilled encoder plus a compressed gallery of a few thousand species fits on a phone, so identification works in forests and fields without signal; the location prior can be shipped per state.
- **More regional species.** Because species are added by appending photos and traits, a state forest department or a university can maintain its own flora file.
- **Expert feedback loop.** When a botanist confirms or corrects an answer in the app, the photo is added to the gallery for that species, so the system improves with use, still without retraining.

## Conclusion

VanaLens identifies 607 common Indian and worldwide plant species from phone photos with calibrated confidence and visible evidence. On {{d:n_photos_vl}} test photos taken after its encoders' training data, it reaches {{rowci:vl600:+ geo prior:top1}} top-1 from one photo with location and {{rowci:vl600:VanaLens up to 3 photos + geo:top1}} from up to three photos of the same plant, meeting the 95% target when the user adds photos. On the same photos it is {{d:h2h_gain_zs}} above BioCLIP 2 zero-shot over the same names and, with up to three photos and location, {{d:h2h_gain_best3}} above a linear probe on the same features. Its confidence is calibrated (ECE {{pct:L3/ablation_vl600.json:rows.ensemble.ece}}), it backs off to genus or family when unsure and is then right {{row:vl600:+ geo prior:backoff_acc}} of the time at the level it answers, it can say a plant may not be in its catalogue, and it learns a new species from five photos in seconds ({{pct:L6/final.json:extras.vl600.few_shot.2.top1_new}} top-1 on 61 unseen species) without retraining anything.

These results come from a deliberate design: frozen foundation features with light trained heads, fused and calibrated scorers, and decisions built on calibrated probabilities. That design is what makes the system cheap to run, quick to extend and able to show real reference photos as evidence. Section 8 sets out the current scope (same-source test photos, a baseline-strength open-set check, calibration-to-test shift, catalogue size, evaluation breadth and unvalidated explanations), each with a concrete next step, and none of those steps requires giving up the frozen-feature core.

## References {-}

### Related-work references (codes used in the text) {-}

Each entry was checked against its publisher, arXiv or dataset page in November 2026, except the canonical works marked (c), which are cited from their standard bibliographic record.

- **R1** Kumar, N., Belhumeur, P. N., Biswas, A., Jacobs, D. W., Kress, W. J., Lopez, I. C. and Soares, J. V. B. Leafsnap: A Computer Vision System for Automatic Plant Species Identification. ECCV 2012, pp. 502-516. doi:10.1007/978-3-642-33709-3_36.
- **R2** Garcin, C., Joly, A., Bonnet, P., Lombardo, J.-C., Affouard, A. et al. Pl@ntNet-300K: a plant image dataset with high label ambiguity and a long-tailed distribution. NeurIPS Datasets and Benchmarks 2021. Model results: [github.com/plantnet/PlantNet-300K](https://github.com/plantnet/PlantNet-300K) (README, test top-1).
- **R3** Kavitha, S., Satish Kumar, T., Naresh, E. and Kalmani, V. H. Medicinal Plant Identification in Real-Time Using Deep Learning Model. SN Computer Science, 2024. doi:10.1007/s42979-023-02398-5.
- **R4** Pushpa, B. R. and Shobha Rani, N. DIMPSAR: Dataset for Indian medicinal plant species analysis and recognition. Data in Brief 49, 109388, 2023. doi:10.1016/j.dib.2023.109388.
- **R5** Sarma, P., Boruah, P. A. and Buragohain, R. MED 117: A dataset of medicinal plants mostly found in Assam with their leaf images, segmented leaf frames and name table. Data in Brief 47, 108983, 2023.
- **R6** HerbLife: leaf images of 112 medicinal plant species, Kottayam, Kerala. Mendeley Data, 2023. [data.mendeley.com/datasets/v8n2twxxdb](https://data.mendeley.com/datasets/v8n2twxxdb).
- **R7** Joly, A., Goeau, H., Bonnet, P. et al. Interactive plant identification based on social image data. Ecological Informatics 23, 22-34, 2014. doi:10.1016/j.ecoinf.2013.07.006.
- **R8** Rzanny, M. et al. Flowers, leaves or both? How to obtain suitable images for automated plant identification. Plant Methods 15, 2019. doi:10.1186/s13007-019-0462-4.
- **R9** Gu, J., Stevens, S., Campolongo, E. G. et al. BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning. NeurIPS 2025 (spotlight). arXiv:2505.23883.
- **R10** (c) Stevens, S. et al. BioCLIP: A Vision Foundation Model for the Tree of Life. CVPR 2024.
- **R11** (c) Radford, A. et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021.
- **R12** (c) Oquab, M. et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR 2024. Simeoni, O. et al. DINOv3. arXiv:2508.10104, 2025.
- **R13** Gustineli, M., Miyaguchi, A., Cheung, A. and Khattak, D. Tile-Based ViT Inference with Visual-Cluster Priors for Zero-Shot Multi-Species Plant Identification. arXiv:2507.06093, 2025 (2nd place, PlantCLEF 2025). Overviews: Goeau, H., Espitalier, V., Bonnet, P. et al. Overview of PlantCLEF 2024, CLEF 2024, arXiv:2509.15768; Martellucci, G., Goeau, H., Bonnet, P. et al. Overview of PlantCLEF 2025, CLEF 2025, arXiv:2509.17602.
- **R14** Maruf, M., Daw, A., Mehrab, K. S. et al. VLM4Bio: A Benchmark Dataset to Evaluate Pretrained Vision-Language Models for Trait Discovery from Biological Images. NeurIPS Datasets and Benchmarks 2024. arXiv:2408.16176 (numbers from its Table 2).
- **R15** (c) Vovk, V., Gammerman, A. and Shafer, G. Algorithmic Learning in a Random World. Springer, 2005. Angelopoulos, A. N. and Bates, S. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511, 2021. Romano, Y., Sesia, M. and Candes, E. Classification with Valid and Adaptive Coverage. NeurIPS 2020.
- **R16** Ding, T., Fermanian, J.-B. and Salmon, J. Conformal Prediction for Long-Tailed Classification. arXiv:2507.06867, 2025.
- **R17** Fermanian, J.-B., Hebiri, M. and Salmon, J. Class conditional conformal prediction for multiple inputs by p-value aggregation. NeurIPS 2025. arXiv:2507.07150.
- **R18** Mac Aodha, O., Cole, E. and Perona, P. Presence-Only Geographical Priors for Fine-Grained Image Classification. ICCV 2019, pp. 9596-9606.
- **R19** Cole, E., Van Horn, G., Lange, C., Shepard, A., Leary, P., Perona, P., Loarie, S. and Mac Aodha, O. Spatial Implicit Neural Representations for Global-Scale Species Mapping. ICML 2023. arXiv:2306.02564.
- **R20** (c) Scheirer, W. J. et al. Toward Open Set Recognition. IEEE TPAMI 35(7), 2013. Hendrycks, D. and Gimpel, K. A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. ICLR 2017. Vaze, S., Han, K., Vedaldi, A. and Zisserman, A. Open-Set Recognition: A Good Closed-Set Classifier Is All You Need. ICLR 2022.
- **R21** (c) Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. On Calibration of Modern Neural Networks. ICML 2017.
- **R22** (c) Menon, S. and Vondrick, C. Visual Classification via Description from Large Language Models. ICLR 2023.
- **R23** (c) Van Horn, G. et al. The iNaturalist Species Classification and Detection Dataset. CVPR 2018.
- Data services: iNaturalist API v1 ([api.inaturalist.org/v1/docs](https://api.inaturalist.org/v1/docs)), iNaturalist Open Data on AWS ([registry.opendata.aws/inaturalist-open-data](https://registry.opendata.aws/inaturalist-open-data)), GBIF occurrence API ([techdocs.gbif.org/en/openapi](https://techdocs.gbif.org/en/openapi/)).
- Products (no technical paper): Google Lens ([lens.google](https://lens.google)), iNaturalist Seek ([inaturalist.org/pages/seek_app](https://www.inaturalist.org/pages/seek_app)).
