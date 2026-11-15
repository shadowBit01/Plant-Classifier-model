# VanaLens pitch script (4 minutes)

Timing: about 25 seconds per slide, 60 seconds for the demo. Numbers come from `runs/L6/final.json` and
`runs/L3/ablation_inat_india.json`. If a number changes, rebuild the deck with `python scripts/build_deck.py`
and update this file to match.

## Slide 1 (0:00 to 0:25) Problem
Point your phone at a plant and any app gives you one name, with confidence, every time. That is a problem when
the plant is a medicinal herb and its look-alike is an adulterant. And if your local species is not in the app,
the only fix is to collect data and retrain a model. We wanted plant ID that tells you when it is not sure.

## Slide 2 (0:25 to 0:50) Why current approaches fall short
Fine-tuned classifiers are closed-set. Their scores are not calibrated, so 90 percent rarely means 90 percent.
And open-ended naming is hard: on our own test photos, BioCLIP 2 naming a plant from the whole tree of life gets
70.4 percent. Give it our 52 species to choose from and it gets 93.5 percent. So we keep strong frozen models,
narrow the choice, and add honesty on top. No fine-tuning.

## Slide 3 (0:50 to 1:15) Our system
Photos go through two frozen encoders, BioCLIP 2 and DINO. Three scorers vote: a linear probe, text prototypes
built from written plant traits, and a nearest-neighbour search that returns real reference photos as evidence.
We fuse them, calibrate with a temperature, apply a location prior, back off to genus when unsure, and return a
conformal set with a 90 percent coverage target. Adding a species is just adding photos.

## Slide 4 (1:15 to 2:15) Live demo
Switch to the app (steps in `pitch/demo.md`). Say while it runs: "One photo in. Out comes the species, the set
we are 90 percent sure about, and the three reference photos it matched. Now a second photo of the same plant,
both are fused. Now a species it has never seen: name, genus, family, a few photos. No training run." It takes
about one second per photo on a laptop CPU.

## Slide 5 (2:15 to 2:40) Results
The test set is 558 iNaturalist photos of 52 Madhya Pradesh species, all taken from July 2025 on, after BioCLIP 2's
training data was collected. VanaLens gets 95.7 percent top-1 and 94.3 percent macro top-1. The best baseline we
ran is 93.5. Be upfront: we could not run the Pl@ntNet API or a VLM baseline without API keys.

## Slide 6 (2:40 to 3:05) Ablation
Text traits alone get about 50 percent with zero photos, and fused with the probe they lift it from 92.1 to 95.3.
The fitted weight on kNN is zero: kNN is there for evidence, not votes. The geo prior changes nothing here because
all 52 species are already local. Multi-photo gain, the reranker and the few-shot curve were not measured.

## Slide 7 (3:05 to 3:30) Clean test, honest confidence
Temperature scaling cuts calibration error from 0.107 to 0.023. Conformal sets hit 95.3 percent coverage
against a 90 percent target. Be honest: every set had one species on this test, so coverage equals top-1 here.
6.6 percent of photos back off to "unknown" instead of a forced name. We did not collect campus photos.

## Slide 8 (3:30 to 4:00) Impact and roadmap
Three uses: herb quality checks where a pharmacist can verify the evidence photos, biodiversity surveys where
anyone can add a local species in minutes, and an offline phone version distilled to a small CNN. Next we switch
on the VLM reranker and measure multi-photo and campus tests. VanaLens: 95.7 percent top-1, honest confidence,
no fine-tuning. Thank you.
