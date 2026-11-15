# Skeptical review of the VanaLens report and deck (card L7-d)

Reviewed: `pitch/report/VanaLens_report.md` (built from `runs/` after the DINO fix, state 2026-11-16 00:34),
the 15 slide PNGs in `pitch/preview/` and `runs/L6/final.json`. The reviewer's stance: assume every claim is wrong
until a number in `runs/` supports it. Each item says what was wrong, then what was done:
**Fixed** (wording, numbers, CI, figure or layout changed), **Limitation** (moved to report Section 8, cannot be fixed
from existing results) or **Needs** (another lane must act).

New checks run for this review: `pitch/report/review_stats.py` (geographic share, test photos per species, same-plant
proxy, from vl600 `meta.csv`) and `pitch/demo_pick.py` (demo photos and an unknown-orchid check on cached
embeddings). Both write small JSON files next to them; the report reads them at build time.

## 1. Claims without numbers, or numbers that contradicted the text

| # | Where | Problem | Action |
|---|---|---|---|
| 1.1 | Report 7.2 ablation | Said temperature "cuts calibration error from 0.6% to 0.6%". After the re-run the fused scores were already calibrated; temperature changed nothing. | **Fixed**: sentence is now generated from the ablation (`d:temp_note`) and says it changes neither top-1 nor ECE. Deck slide 11 notes said "raw fused scores are overconfident"; rewritten. |
| 1.2 | Report 7.5, deck 10 | Selective answering "reaches 94.4%, below target because test is harder". In fact the calibration threshold answers 100% of test photos, so the step does nothing. Deck said "abstaining adds little: 94.4% on 100.00% answered". | **Fixed**: text states the step is inactive on vl600; deck bullet says "abstaining did nothing (100% answered)". **Limitation** 17. |
| 1.3 | Report 7.2, Limitation 1 | kNN note promised "the cleanest test of whether DINO adds information" but gave no number; Limitation 1 said the DINO benefit "rests on a single re-run". The L6-e baselines now measure it: BioCLIP 2-only probe 93.2%, DINOv2-only 88.8%, both 93.4% (+0.2 points, intervals overlap). | **Fixed**: both sentences generated from the baselines block (`dino_note`). Deck slide 14 says DINOv2 adds +0.2 points. **Limitation** 1 rewritten. |
| 1.4 | Report 6.3 | "Baseline numbers ... are point values": no longer true, L6-e baselines carry CIs on the same resamples. | **Fixed**. |
| 1.5 | Report abstract, conclusion | Described the DINO loader bug as current ("kept the second encoder from contributing"). It was fixed before these numbers. | **Fixed**: now "a second encoder that adds little even after a late loader fix". |
| 1.6 | Deck 2 | "Apps give false confidence" and "apps return one confident name, even for unknown plants": no measurement supports this (we never ran an app). | **Fixed**: "Apps name plants but rarely say how sure they are"; "none we found says 'not in our list'", matching report Section 2. |
| 1.7 | Report 2 | Two overlapping reference lists (numbered 1-18 and R1-R23) with mixed citation styles ("Menon and Vondrick, 2023", "Ding et al., 2025"). | **Fixed**: one R-coded list; inline citations converted; data services listed. |
| 1.8 | Report derived text | Cross-references pointed to "Section 3.11" and "Section 6.6" (bug is 4.11, open-set is 7.6). | **Fixed** in `build_report.py`. |

## 2. Missing confidence intervals

| # | Where | Problem | Action |
|---|---|---|---|
| 2.1 | Report 7.9 live pipeline | 97.0% top-1 on only 100 observations, no interval; reads like a second accuracy estimate. | **Fixed**: Wilson 95% CI 91.5 to 99.0 added, and the text says it only shows that live and cached agree. **Limitation** 18. |
| 2.2 | Report 2.4, deck 9 | "+0.6 points over the best baseline" with overlapping CIs. Per-method CIs do not test a difference; no paired bootstrap was run. Deck said "not significant", which is also untested. | **Fixed**: report says overlap is not a test of equality; deck says "intervals overlap". **Limitation** 15. |
| 2.3 | BioCLIP 2 zero-shot rows | Point values only (no per-photo predictions). | **Limitation** (already disclosed in table captions). |
| 2.4 | Location prior | +0.3 points on vl600, -0.2 on inat_india, intervals with and without the prior overlap on both; text treated it as a feature. | **Fixed**: report 7.4 now says it is a small effect; **Limitation** 9 gives the numbers. Demo shows one case where it matters, labelled as such. |

## 3. Training-data contamination

| # | Problem | Action |
|---|---|---|
| 3.1 | Goal 1 said test photos are ones "the underlying models cannot have seen"; the date cut is evidence, not proof (DINOv2's LVD-142M is web-crawled). | **Fixed**: "taken after the encoders' training data was collected". Deck slide 1 likewise. |
| 3.2 | The date cut keeps photos apart, not plants. 82 of 9,094 test observations with GPS (0.9%) share species and rounded coordinates (about 11 m) with a gallery or calibration observation: probably the same tree photographed again. | **Limitation** 14 (number computed by `review_stats.py`). |
| 3.3 | BioCLIP 2 has seen many pre-cutoff photos of every vl600 species, so the test measures new photos of familiar species. On inat_india BioCLIP 2 zero-shot alone gets 93.5% and the text scorer alone matches the full system. | **Fixed**: stated in report 7.2 (inat_india paragraph) and **Limitation** 14. |
| 3.4 | Trait texts were written by an LLM that may have seen iNaturalist descriptions. | **Limitation** 7 (already). |

## 4. Dataset and geographic bias

| # | Problem | Action |
|---|---|---|
| 4.1 | Species chosen by iNaturalist observation counts favour showy, urban, roadside and cultivated plants. | **Limitation** 13. |
| 4.2 | Only 70.1% of vl600 test observations are from India; the rest are worldwide fills. "India-focused" results are not reported by region. | **Limitation** 13; deck slide 7 now says "70% of test from India". |
| 4.3 | Gallery and test come from the same source (iNaturalist research grade); no own-phone test. | **Limitation** 5 (already); added to deck slide 14. |

## 5. Few test photos per species

| # | Problem | Action |
|---|---|---|
| 5.1 | 9 to 15 test observations (15 to 35 photos) per species: per-species accuracy is too noisy to report; only pooled metrics are reliable. | **Limitation** 12. |
| 5.2 | inat_india open-set (AUROC 0.97) and few-shot (100% at 5 shots) rest on 5 held-out species and 53 test photos. | **Limitation** 3 and 12; the report already leads with vl600. |

## 6. Baselines not run, unfair comparisons

| # | Problem | Action |
|---|---|---|
| 6.1 | Pl@ntNet API (B0), general VLM (B4) and the VLM reranker not run (no keys). | **Limitation** 6 (already). |
| 6.2 | Deck slide 8 compared "VanaLens, 1 photo" (which used GPS) with baselines that had no location. | **Fixed**: label "1 photo + location"; title now "Ahead of BioCLIP 2 zero-shot" (not "beats BioCLIP 2"). Slide 9 is the like-for-like comparison. |
| 6.3 | The headline gain over BioCLIP 2 zero-shot (+6.8 points) is expected by design: zero-shot sees no gallery photos. The fair baseline is the probe on the same features (+0.6). | **Fixed**: said in report 2.4, deck slide 6 now gives both numbers, slide 9 title "level with a probe". |
| 6.4 | "In final.json the B2 row is labelled kNN" but the script runs zero-shot. | Already disclosed in report 6.4. **Needs** L6: rename the row in `scripts/final_table.py` (cosmetic). |

## 7. Overclaiming

| # | Where | Problem | Action |
|---|---|---|---|
| 7.1 | Deck 4, 5 | "Get a species, a short list, or not in catalog"; pipeline box "Not sure: short list, 90% set". The 90% sets have mean size 1.0 on both datasets: there is never a short list. | **Fixed**: "a genus or family when unsure"; slide 3 table cell "90% (size 1)"; slide 11 "Sets are always one species: coverage holds only because top-1 is above 90%". |
| 7.2 | Abstract, deck | 95% target: single-photo top-1 is 94.4% (CI 94.0 to 94.8), so it is missed from one photo. | **Fixed**: abstract and conclusion say so; deck slide 10 bullet "1 photo: 94.4%, just short of 95%". |
| 7.3 | Report contribution 3, deck 6, 13 | "Evidence with every answer" implies the evidence explains the decision. The kNN gallery that supplies the reference photos has fusion weight 0; the app shows reference file names, not images. | **Limitation** 16; deck slide 13 caption says the references are listed by file. **Needs** L5: render evidence thumbnails. |
| 7.4 | Report 7.6, deck 12 | "Unknown check" sounds like a guard against unseen species. New check: 0 of 11 photos of *Vanda tessellata* (an orchid missing from vl600) were rejected; 10 were named as a related orchid with at least 70% confidence. After adding 5 photos, 11 of 11 were named. | **Fixed**: added to report 7.6 as a generated sentence; **Limitation** 3; demo step 2 is honest about it. |
| 7.5 | Conclusion | Implied VanaLens' accuracy comes from the system. | **Fixed**: "Most of its top-1 comes from frozen BioCLIP 2 features with a linear head; what VanaLens adds is mainly capability and honesty." |

## 8. Unclear explanations

| # | Problem | Action |
|---|---|---|
| 8.1 | Report 3 step 3 says DINO captures "fine visual detail (texture, venation, margins)"; nothing measures that, and DINO adds +0.2 points. | Left as design intent; the measured contribution is now stated next to it in 7.2 and Limitation 1. |
| 8.2 | Ensemble chosen over three-scorer fusion on calibration, but the ensemble has higher test ECE (0.9% vs 0.6%). | Visible in the ablation table and slide 11; no change needed. |
| 8.3 | The app's "not sure" card fires whenever the top probability is below 0.967, because `app/app.py` `_get_confidence_threshold` reads `state["thr"]`, which is the open-set similarity threshold, not a probability. Correct answers at 83% are shown as "not sure". | **Needs** L5 (app is not L7's). Demo text warns about it. |

## 9. Slide design

| # | Problem | Action |
|---|---|---|
| 9.1 | Titles wrapped to two lines on slides 2, 3, 6, 8, 9, crowding the content. | **Fixed**: all titles are one line. |
| 9.2 | Slide 10 chart on a 0-100 axis made 94.0, 94.4 and 96.0 look identical, so the 95% line told nothing. | **Fixed**: axis starts at 85 and the axis label says so; value labels have a white backing so the target line does not cross them. |
| 9.3 | Slide 11 ECE chart on a 0-17% axis with values near 1%: bars were slivers. | **Fixed**: axis scales to the data (0 to about 2%). |
| 9.4 | Slide 14 had large empty space and omitted the two most important gaps (second encoder, same-source test). | **Fixed**: four bullets now include both. |
| 9.5 | Slide 13 app screenshot shows file names as "nearest references" and an en dash in the app title. | Caption fixed; the screenshot belongs to L5's app. |
| 9.6 | Slides 8 and 9 overlap in content. | Kept both (8 is the simple story, 9 the fair one); 8 now labels the GPS input. |

## 10. What remains open (owner)

- L5: evidence thumbnails in the result card; use a probability threshold (for example the backoff tau 0.6) instead of
  `state["thr"]` in `_get_confidence_threshold`.
- L6: store per-photo predictions so a paired bootstrap of differences and per-species tables become possible;
  rename the "B2 BioCLIP 2 kNN" row.
- Everyone: a test set of our own phone photos (campus or market) is the single most useful missing experiment.
