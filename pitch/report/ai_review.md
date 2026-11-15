# Strict AI-reviewer self-check of the VanaLens report and deck (card L7-f)

Read: `pitch/report/VanaLens_report.pdf` (22 pages, built from `runs/`), `pitch/preview/slide_01..15.png` and the
speaker notes in `pitch/script.md`. Stance: an AI assistant asked "is it good, what is novel, what are the
trade-offs and drawbacks", which checks every claim against a number and flags overclaiming first.

## Pass 1

### Strengths it would list
- Every headline number has a 95% bootstrap CI over observations; baselines share the same resamples.
- Test photos are all post-cutoff (on or after 1 July 2025), which limits overlap with BioCLIP 2's training data.
- Head-to-head on the same 12,831 photos, same features and splits: +6.8 points over BioCLIP 2 zero-shot
  (one photo, no location), +2.6 over a linear probe on the same features with up to three photos and location.
- 96.0% (95.7 to 96.4) top-1 with up to three photos; ECE 0.9%; backoff 97.1% correct at the level it answers.
- Add-species without retraining is measured: 80.9% top-1 on 61 unseen species from five photos each.
- Prior-work numbers are kept on their own datasets and labelled NOT comparable; no state-of-the-art claim.
- Weight-integrity tests on the encoder loader (`tests/test_l2_weights.py`).

### Novelty it would credit
- A systems contribution: calibrated fusion, conformal sets, hierarchical backoff, open-set flag, GBIF prior,
  multi-photo fusion and gallery evidence in one pipeline on frozen features, evaluated jointly.
- vl600: an India-focused benchmark with a date cut, 607 species by observation count.
- It would not credit a new model or encoder (and the report does not claim one).

### Drawbacks it would raise, and what was done
| # | Drawback | Action |
|---|---|---|
| 1 | Abstract said "unchanged" for known-species accuracy after adding species (94.2% before, 94.0% after). | Fixed: "kept at 94.0%". |
| 2 | "96.0% top-1" in the subtitle and slide 4 title without saying it needs up to 3 photos. | Fixed: subtitle says "with up to three photos"; slide 4 title "96.0% on 3 photos". Summary also states one photo is just below the 95% target. |
| 3 | +2.6 vs the probe compares 3 photos + location against a 1-photo probe. | Wording: every place that quotes +2.6 states the input (up to 3 photos + location); the summary, Section 2.4 and slides 6 and 9 also give the 1-photo, no-location gain (+0.6, intervals overlap). |
| 4 | Head-to-head slide title said "level with a probe" in one place and the report said "+0.6": inconsistent tone. | Fixed: slide title now gives both numbers; notes say the single-photo gain alone is not statistically clear. |
| 5 | Trade-off of frozen encoders not explained. | Fixed: Section 4.1 lists the benefits and the cost (features not specialised to these species), with a fine-tuned reference row as a next step. |
| 6 | "Seconds" for adding a species not tied to a number. | Fixed: tied to the measured 4.6 s per observation of up to 3 photos on CPU (encoding is the only cost; no gradient step). |
| 7 | Few-shot text prompt approximation not mentioned near the 80.9% result. | Fixed: stated in Section 7.7, with the inat_india 5-species caveat. |
| 8 | Old limitation list (18 items) mixed resolved items and pre-fix numbers. | Fixed: one paired table of 6 scope items with next steps; resolved loader issue described only as a safeguard in 4.13; no pre-fix numbers anywhere. |

## Pass 2 (after the fixes)

Report and deck numbers agree (checked: 94.4, 96.0, 98.5, +6.8, +0.6, +2.6, 0.9% ECE, 97.1, 80.9, 38%/6%,
AUROC 85.2, 4.6 s). No em dashes, no hidden or tiny text, no text addressed to reviewers.

### Drawbacks it would still raise (material, cannot be fixed by wording)
- Test photos come from the same source (iNaturalist) as the gallery; no out-of-source phone-photo test.
- No paired bootstrap of the VanaLens-minus-probe difference (per-photo predictions not stored), so the
  single-photo +0.6 gain is not shown to be significant.
- Open-set flag is baseline strength (AUROC 85.2%, 38% of unseen species flagged at 6% false flags).
- Conformal sets are almost always size 1 and the selective step answers 100% of test photos (calibration-to-test shift).
- Pl@ntNet API and VLM baselines not run (no keys); traits generated with a language model, pending botanist review.
- DINOv2 adds only +0.2 points over BioCLIP 2 alone.

All six are stated in report Section 8 with a concrete next step, and summarised on slide 14.
