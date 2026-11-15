# VanaLens live demo (about 3 minutes, 5 steps)

Every photo below was chosen by `pitch/demo_pick.py`, which runs the app's own scoring code on the cached vl600
embeddings (CPU, no network) and stores what it expects in `pitch/demo_picks.json`. The live app re-encodes the photo
files, so rehearse once: if a live answer differs from the "expected" line, pick the next candidate in that file.
Photos are iNaturalist observations (CC BY or CC BY-NC); show the observation number if asked.

## Before the pitch (15 minutes early, with Wi-Fi)

1. PowerShell, repo root. Use the 607-species model, not the default 52-species one:
   `$env:VANALENS_DATASET = 'vl600'; & $env:VANALENS_PY -m app.app`. Do NOT set `VANALENS_MOCK`.
2. Open the local URL it prints (usually http://127.0.0.1:7860).
3. Copy the demo photos to `Desktop\demo_photos` (copies only, nothing is moved). `$D` is `$env:VANALENS_DATA`
   (default `~\vanalens-data`):

   | File in demo_photos | Copy from `$D\...` |
   |---|---|
   | `1_known.jpg` | `vl600\Basella alba\340742386_620140577.jpg` |
   | `2_orchid.jpg` | `inat_india\Vanda tessellata\339161830.jpg` |
   | `2_orchid_add\` (5 photos) | `inat_india\Vanda tessellata\` 194194825, 33846405, 126939195, 134378757, 224832509 (`.jpg`) |
   | `3_photo_a.jpg`, `3_photo_b.jpg`, `3_photo_c.jpg` | `vl600\Albizia lebbeck\` 372869981_681391420, 372869981_681392318, 372869981_681392328 (`.jpg`) |
   | `4_gps.jpg` | `vl600\Azadirachta indica\296700879_534275682.jpg` |

4. Warm up: identify `1_known.jpg` once (the first call loads the models). After that one observation takes about
   4.6 s on the laptop CPU (measured over 100 vl600 observations, runs/L6/e2e_vl600.json).
5. Warm the location cache: identify `4_gps.jpg` once with latitude 11.4904, longitude 77.5611. Its GBIF cell is
   then stored in `$VANALENS_CACHE\gbif`, so step 4 works offline.
6. Rehearse steps 1 to 4 once, take a screenshot of every result card, and record the whole run to
   `demo_photos\backup.mp4`.

## On stage

1. **A known plant (20 s).** Tab "Identify". Upload `1_known.jpg` (Malabar spinach), leave location empty, Identify.
   Expected: *Basella alba* at species level, confidence close to 100%, a one-species 90% set, matched traits and the
   nearest gallery photos. Say: "a name, how sure it is, and why".
2. **An unknown plant, then teach it (50 s).** Upload `2_orchid.jpg`, a *Vanda tessellata* orchid. It is not one
   of the 607 species. Expected: a "not sure" card whose best guess is a different orchid (*Rhynchostylis retusa*,
   about 72%). Say honestly: "It does not know this orchid. Its unknown check did not fire here; for close relatives
   of known species it usually does not, which is the part with the most room to improve. But it did not claim to be sure."
   Then tab "Add a species": name *Vanda tessellata*, genus *Vanda*, family *Orchidaceae*, upload the 5 photos in
   `2_orchid_add\`, Add. Back to "Identify", upload `2_orchid.jpg` again. Expected: *Vanda tessellata*, about 98%.
   Say: "five photos, no retraining, a few seconds".
3. **A plant from three photos (35 s).** Upload only `3_photo_a.jpg` (*Albizia lebbeck*, the siris tree). Expected:
   a wrong, unsure guess (*Millingtonia hortensis*, about 53%), shown as "not sure". Now upload all three
   `3_photo_*.jpg` together. Expected: *Albizia lebbeck* now leads, about 83% (the card still says "not sure": the app
   shows a sure answer only above about 97%). Say: "one photo can mislead; three photos of
   the same plant are combined, and on the whole test set this lifts top-1 from 94.4% to 96.0%".
4. **Location (30 s).** Upload `4_gps.jpg` (a neem branch) with location empty. Expected: *Robinia pseudoacacia*
   (black locust, a North American tree), about 62%. Enter latitude 11.4904, longitude 77.5611 (Tamil Nadu, where the
   photo was taken) and Identify again. Expected: *Azadirachta indica* (neem) now leads, about 76%, still on a "not sure" card. Say: "the location prior
   knows black locust is rare here; on the full test set it is a small effect, +0.3 points, but it fixes cases like this".
5. **Close (5 s).** "Every answer has a confidence you can trust, evidence you can check, and an honest 'not sure'."

The expected numbers above come from the cached run; the live percentages may differ by a few points. If the
location in step 4 does not change the answer live, say what the cached run showed and move on.

## If something fails

- **Wi-Fi fails.** The app, the models, the gallery and the GBIF cell for step 4 are all on the laptop, so steps 1 to
  4 run offline. Do not type a new location while offline: an uncached location waits about 10 s for GBIF, then
  silently answers without the location prior. The optional VLM reranker needs the network, but it is off in this
  build anyway.
- **The app will not start or hangs.** Play `demo_photos\backup.mp4` and narrate the same steps, or show the
  screenshots. Slide 13 of the deck has a real result card as a last resort.
- **A live answer differs from the expected one.** Say what it shows (the card always shows its confidence and set)
  and move on; that is the point of a calibrated system. Before the pitch, swap in the next candidate from
  `pitch/demo_picks.json` (keys `known`, `geo`, `multi`).
- **Projector issue.** Describe the demo in two sentences from slide 13 and continue with slide 14.
