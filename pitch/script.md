# VanaLens: talk script (about 4.5 minutes)

Matches `pitch/VanaLens.pptx` (12 slides). Every number comes from `runs/` (mainly `runs/L6/final.json`, vl600 test split).
Spoken pace is about 140 words per minute. Times are cumulative.

## 1. Title (0:00 to 0:15)
Hi, we are presenting VanaLens. It names common plants from a phone photo, and it tells you honestly when it is not sure.
Everything you will see was measured on photos taken after our models were trained.

## 2. The problem (0:15 to 0:40)
Everyone walks past plants they cannot name: a tree on the street, a herb in the market, a weed in a field.
Plant apps exist, but they give one confident answer, even for a plant they have never seen.
Asked an open question, BioCLIP 2, a strong biology model, names only 62% of our test photos.
And if your local plant is missing, adding it usually means retraining.

## 3. What we built (0:40 to 1:05)
VanaLens names 607 common plants. You upload one to three photos and get a species, a short list, or "not in catalog".
Three ideas: no retraining, because the vision models stay frozen and a new species is just a few photos;
honest uncertainty, with calibrated confidence and 90% prediction sets; and context, from extra photos and location.

## 4. How it works (1:05 to 1:35)
A photo goes through two frozen encoders, BioCLIP 2 and DINOv3.
Three scorers read the embeddings: a linear probe, a nearest-neighbour search that also gives us evidence photos,
and text prototypes built from written plant traits.
We fuse them, calibrate with one temperature, and multiply in a location prior from GBIF records.
A conformal rule then decides: one answer, a short list, or not in catalog.

## 5. Dataset (1:35 to 2:00)
We built our own dataset, vl600: 607 species, 61,367 photos. 400 are the most observed plants in India, 200 the most
observed worldwide, plus 7 common cultivated plants. Together they cover 71.6% of India's research-grade iNaturalist plant
records. All 12,831 test photos were taken after July 2025, after BioCLIP 2's training data, so the test is clean.

## 6. Results (2:00 to 2:35)
On those 12,831 photos, BioCLIP 2 zero-shot gets 87.3% when we give it our 607 names. A linear probe on our embeddings
gets 92.4%. VanaLens gets 93.9% from one photo, and 95.8% with up to three photos of the same plant.
The error bars are 95% bootstrap intervals. On our second test set, inat_india, it is 96.8% against 93.5%.
We could not run the Pl@ntNet API or a VLM baseline, because we had no API keys.

## 7. More photos (2:35 to 3:00)
One photo gives 93.3%. Adding location lifts it to 93.9%. Up to three photos lifts it to 95.8%, and the whole
interval, 95.4 to 96.2, sits above our 95% target. When it is unsure it can back off to genus or family, and the answer
it reports is then right 96.8% of the time. Simply abstaining barely helps on this data.

## 8. Honest uncertainty (3:00 to 3:30)
Raw fused scores are overconfident, with a 9.3% calibration error. One temperature, fitted on a separate calibration
split, brings it to 0.7%. So when VanaLens says 90%, it is right about 90% of the time.
The 90% conformal sets contained the right species 93.3% of the time. Honestly, because accuracy is already above 90%,
those sets are almost always a single species.

## 9. Unknown plants and adding species (3:30 to 4:00)
We hid 61 species to test unknown plants. VanaLens flagged 52% of them as not in catalog, but also wrongly flagged 24% of
known photos, so this is our weakest part. Then we added those species back with a few photos each and no retraining:
78% correct from 5 photos, 84% from 10, while the old species stay at 93.3%.

## 10. The app (4:00 to 4:15)
This is the app. Each answer shows the species, a calibrated confidence, the 90% set, the traits that matched, and the
nearest reference photos as evidence. A second tab adds a new species.

## 11. Limitations (4:15 to 4:35)
To be upfront: the unknown-plant check needs work, the sets rarely grow beyond one species, two baselines are missing,
and the live app needs the gated DINOv3 weights; without them it falls back to DINOv2 and over-flags "not in catalog".

## 12. Thank you (4:35 to 4:45)
VanaLens: 93.9% from one photo, 95.8% from three, honest about its doubt, and new species from five photos.
Thank you, we are happy to take questions.
