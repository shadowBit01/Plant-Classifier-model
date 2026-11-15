# VanaLens live demo (60 seconds)

## Before the pitch (do this 15 minutes early)
1. Plug in the laptop. In the repo root (PowerShell): `& $env:VANALENS_PY -m app.app`. Do NOT set `VANALENS_MOCK`.
2. Open the local URL it prints (usually http://127.0.0.1:7860) in the browser. It runs locally, so it needs no Wi-Fi.
3. Warm it up: identify one photo once. The first call loads the models (about 70 s in our e2e run, cold); after
   that it takes about 1 s per photo on CPU.
4. Put the demo photos in one folder on the desktop, `demo_photos/`, copied from `$VANALENS_DATA` (inat_india test split):
   - `1_leaf.jpg`: a leaf of a species from the test split (for example Calotropis procera).
   - `2_flower.jpg`: a flower photo of the same species.
   - `new_species/`: 3 to 10 photos of a plant that is NOT in the 52 classes, plus its name, genus and family on a sticky note.
5. Keep a backup screen recording of steps 1 to 3 below at `demo_photos/backup.mp4`, and screenshots of each result.

## On stage
1. **Identify** (20 s). Tab "Identify". Upload `1_leaf.jpg`, click Identify.
   Point at: the species label and confidence, the "90% sure it's one of these" chips, the 3 evidence photos and matched traits.
2. **Second photo** (15 s). Upload `1_leaf.jpg` and `2_flower.jpg` together, click Identify.
   Point at: one fused answer for both photos. If the first result showed a hint asking for a flower photo, say so.
3. **Add a species** (25 s). Tab "Add a species". Type the name, genus, family; upload the `new_species/` photos; click Add.
   Go back to "Identify", upload one more photo of that plant, show that it is now recognised. Say: "no retraining".

## If something fails
- **Wi-Fi down**: nothing changes. The app and models run on the laptop. Only the optional VLM reranker would need
  the network, and it is off in this build.
- **App will not start or hangs**: play `demo_photos/backup.mp4` and narrate the same three steps.
- **Projector or laptop issue**: skip to slide 5 and describe the demo in one sentence from slide 4.
- **Wrong answer on stage**: point at the set and the evidence photos. That is the point of the system:
  you can see why it answered and how sure it is.
