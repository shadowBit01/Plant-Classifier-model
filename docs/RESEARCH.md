# Plant Species Classifier: Audit, Research, Baselines and 24h Build Plan

Compiled 15 Nov 2026. Input: a JPEG/PNG photo (leaf, flower, whole plant, bark, fruit). Output: the species, with confidence.

---

## 0. TL;DR

- **Don't make fine-tuning the core.** Frozen biology foundation models already encode most of what a 24h fine-tune would learn. Meta's own DINOv3 model card says to keep fine-tuning as a last resort and use frozen features with simple heads. On a 6 GB GPU with small regional data, fine-tuning costs hours, adds a few points at best, and kills the "add a species without retraining" property.
- **Recommended build: an evidence-backed, uncertainty-aware plant ID system.** Frozen encoders (BioCLIP 2 + DINOv3) → retrieval over a reference gallery + linear probe + text-trait zero-shot → geo prior → hierarchical backoff → conformal prediction set → VLM reranker only when uncertain. Every answer ships with nearest reference photos, matched traits, and an honest "one of these N" set.
- **Add a species without retraining**: 5 photos (few-shot) or just a name + traits (zero-shot).
- **Baselines to beat**: Pl@ntNet API (production), Pl@ntNet-300K published CNN weights, BioCLIP 2 zero-shot, DINOv3 linear probe, a general VLM asked open-ended.
- **Eval sets**: Pl@ntNet-300K (credible benchmark), DIMPSAR Indian medicinal leaves (local story), 15-20 campus species photographed by the team (the only fully clean test set).
- **Hardware fit**: everything runs inference-only on a 6 GB laptop GPU in fp16. Bulk embedding extraction goes to Kaggle (2x T4 or P100, about 30 GPU-h/week).

---

## 1. Audit: what can the model identify from?

| Input | State of the art | Verdict |
|---|---|---|
| Isolated leaf, plain background (scan/lab) | Near-saturated. Flavia (32 species, 1,907 images) and similar lab sets already had recognition rates above 99% by 2014. Leafsnap (2012) got 96.8% top-5 over 184 trees with handcrafted contour features | Too easy to impress alone. Sanity check only |
| Leaf in the wild (phone photo, clutter) | Much harder. Leafsnap keeps 7,719 field images separate from 23,147 lab images for this reason | Core use case |
| Flower | Most informative single view: 88% top-1 vs 77% for whole plant (Rzanny 2019, 101 species) | Support, encourage |
| Whole plant / habit | Weakest single view (77%) | Support, nudge user to add a close-up |
| Bark, fruit | Species-dependent. Pl@ntNet API takes organ tags leaf / flower / fruit / bark | Support via organ router |
| Several photos of one plant | Score fusion: 97% with all 5 views; 96% with flower front + flower side + leaf top | Big, cheap win. Build it |
| Herbarium sheet | Large datasets exist (NAFlora-1M) but domain gap to field photos | Skip unless pitching to herbaria |
| Multi-species scene (vegetation quadrat) | Research frontier: PlantCLEF 2025 best sample-F1 around 0.36 | Moonshot only |

**Decision**: accept 1 to N JPEG/PNG photos of any organ, auto-detect the organ, fuse when there are several.

---

## 2. Approach options

| # | Approach | Training cost | Quality | Novelty | Fits 24h? |
|---|---|---|---|---|---|
| A | Handcrafted leaf shape / venation features (Leafsnap era) | Low | Lab leaves only | None | No |
| B | Fine-tune CNN/ViT end to end | High on 6 GB | Good closed-set; brittle for new species; modest gain over a frozen probe | Low | Weak |
| C | Frozen embeddings + linear probe / kNN | Minutes | Strong | Low/medium | Yes |
| D | BioCLIP 2 zero-shot (open vocabulary, about 952K taxa) | None | Strong on well-photographed species; unknown on Indian endemics | Medium | Yes |
| E | General VLM open-ended ("what plant is this?") | None | Weak: GPT-4V about 17% open-ended in VLM4Bio | Low | Baseline only |
| F | **Hybrid: C + D + geo prior + conformal sets + gated VLM reranker** | Hours | Best + explainable | **High** | **Yes (recommended)** |
| G | Multi-species scene parsing (PlantCLEF-style tiling) | High | Low absolute numbers | High | Moonshot |

### Three build options

1. **Safe (about 6h)**: BioCLIP 2 + DINOv3 embeddings, logistic regression, Gradio demo. Solid accuracy, nothing new to pitch.
2. **Recommended (about 16-18h)**: option F, detailed in Section 6. Same accuracy floor as Safe, plus calibrated sets, visual evidence, few-shot add-a-species, location awareness, multi-photo fusion.
3. **Moonshot**: option G, or herbarium-to-field transfer. Research-grade, high risk of a weak demo.

---

## 3. Datasets (quality-ranked)

### Tier 1: benchmark-grade

| Dataset | Size | Notes | Links |
|---|---|---|---|
| **Pl@ntNet-300K** (NeurIPS 2021 Datasets track) | 306,146 images, 1,081 species. Split 243,916 / 31,118 / 31,112 | Real citizen-science photos, heavy long tail (80% of species hold only 11% of images), many look-alikes. Authors recommend set-valued metrics (macro top-k, average-k). 31.7 GB. **Pretrained ResNet weights in the repo** = instant baseline | [GitHub](https://github.com/plantnet/PlantNet-300K), [Zenodo v1.1](https://zenodo.org/records/5645731), [paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/7e7757b1e12abcb736ab9a754ffb617a-Paper-round2.pdf) |
| **PlantCLEF 2024/2025 training set** | 1,408,033 images, 7,806 species, 1,446 genera, 181 families (south-western Europe flora) | Organizers ship DINOv2 ViT-B/14 models fine-tuned on it: a free plant-specialized encoder | [PlantCLEF 2025](https://www.imageclef.org/PlantCLEF2025), [overview paper](https://arxiv.org/abs/2509.17602), [pretrained models](https://doi.org/10.5281/zenodo.10848263) |
| **iNaturalist 2021** | 10,000 species (all kingdoms; Plantae subset usable), 2.7M train, "mini" 500K (50/class), 100K val | Research-grade labels. Full download about 317 GB, so use mini or the Plantae subset | [GitHub](https://github.com/visipedia/inat_comp/tree/master/2021) |
| **TreeOfLife-200M** | 214M images, 952,257 taxa | CC0 catalog; images are fetched from source sites. Use it to mine a gallery, not to download whole | [Hugging Face](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) |
| **iNaturalist Open Data (AWS)** | 70M+ licensed photos + taxa / observation metadata including coordinates | No AWS account needed. Best source for an India or Jabalpur gallery with GPS | [AWS registry](https://registry.opendata.aws/inaturalist-open-data), [docs](https://github.com/inaturalist/inaturalist-open-data) |

### Tier 2: regional / organ-specific

| Dataset | Size | Notes | Links |
|---|---|---|---|
| **DIMPSAR** (Indian medicinal; Karnataka, Kerala) | 5,900 whole-plant images of 40 species + 6,900 single-leaf images of 80 species | Smartphone, varied backgrounds, lighting and seasons. Best Indian set for this task | [Mendeley](https://data.mendeley.com/datasets/748f8jkphb), [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10375553/) |
| MED117 (Assam) | 117 species, about 77,700 frames cut from leaf videos on white background | Frames are near-duplicates: split by video, never by frame | [Mendeley](https://data.mendeley.com/datasets/dtvbwrhznz/2) |
| HerbLife (Kerala) | 112 medicinal species, leaf images | Kottayam district | [Mendeley](https://data.mendeley.com/datasets/v8n2twxxdb) |
| Medicinal Leaf Dataset (2020) | About 1,500 images, 30 species (listing also says 40), 60-100 per species | Part of BioCLIP's eval suite ("Med. Leaf") | [Mendeley](https://data.mendeley.com/datasets/nnytj2v3n5) |
| SIMPD (South Indian Medicinal Plants, 2025) | Curated, real-world conditions, taxonomy metadata | Data in Brief | [paper](https://www.sciencedirect.com/science/article/pii/S2352340925003907) |
| Rzanny multi-organ (Flora Capture) | 101 species x 100 observations x 5 fixed views | The balanced multi-organ set; ideal for testing view fusion | [paper](https://doi.org/10.1186/s13007-019-0462-4) |
| LeafSnap | 185 north-east US tree species; 23,147 lab + 7,719 field images + segmentations | Classic; lab/field split shows the domain gap | [site](http://leafsnap.com/dataset/) |
| Herbarium 2021 / 2022 (NAFlora-1M) | 2.5M images / 64,500 taxa; 1.05M images / 15,501 species | 2022 Kaggle winner reached 87.66% macro-F with a ~1B-parameter ensemble | [Herbarium 2021](https://www.frontiersin.org/articles/10.3389/fpls.2021.787127), [NAFlora-1M](https://github.com/dpl10/NAFlora-1M), [Kaggle](https://www.kaggle.com/c/herbarium-2022-fgvc9) |

### Data traps

- **Contamination.** BioCLIP 2's TreeOfLife-200M aggregates GBIF, and Pl@ntNet publishes observations to GBIF. The Pl@ntNet API is trained on Pl@ntNet's own data. So BioCLIP 2 zero-shot and B0 numbers on Pl@ntNet-300K test images may be optimistic. Compare everything on DIMPSAR and on your own campus photos too, and say so on the slide.
- **Duplicates.** Small Mendeley leaf sets often contain near-identical shots. Use grouped splits and report cross-dataset results.
- **Label noise.** iNat research-grade labels are estimated at roughly 85-95% correct.

---

## 4. Baselines

| ID | Baseline | How | Published reference point |
|---|---|---|---|
| B0 | **Pl@ntNet API** (production system) | Free tier: 500 identifications/day. Send the organ tag. Run once on about 400 sampled test images, cache results. Unfair on Pl@ntNet-300K (own data), fair on DIMPSAR and campus set | None for your sets: this is the "industry" bar |
| B1 | Pl@ntNet-300K CNNs | Load repo weights (resnet18 / resnet50 etc.), eval only, about 30 min | Paper tables (macro top-k, average-k) |
| B2 | **BioCLIP 2 zero-shot** | `pybioclip`: open-ended TreeOfLife prediction, or `CustomLabelsClassifier` restricted to your label set | BioCLIP 2: +18.1% species classification over BioCLIP. BioCLIP: +17% zero-shot over CLIP on average |
| B3 | DINOv3 ViT-L linear probe | Frozen features + logistic regression | iNat2021, ViT-B/16: 60.9 linear probe vs 68.4 fine-tuned (reported on an aggregator, verify before quoting) |
| B4 | General VLM open-ended | Ask Claude / Gemini for the species name | VLM4Bio: GPT-4V about 17% on open-ended species questions |
| B5 | (Optional) fine-tuned ViT | Only if hours remain. Shows the fine-tune gap is small relative to cost | |

**Metrics** (report all; the long tail matters):

- top-1, top-5, **macro top-1** (mean of per-species accuracy)
- **average set size at 90% coverage** (conformal) and **ECE** (calibration)
- genus- and family-level accuracy ("better mistakes")
- open-set AUROC ("not in our flora" detection)
- few-shot curve: 1 / 5 / 10 reference images per species
- latency per image on the laptop GPU

---

## 5. Research summary

**1. BioCLIP** (CVPR 2024, best student paper). CLIP trained on TreeOfLife-10M (10M images, 450K+ species from iNaturalist, BIOSCAN-1M and EOL) with full taxonomic strings as text. Beats CLIP/OpenCLIP by 17% zero-shot and 16% few-shot on average; the CLIP objective clearly beat cross-entropy and hierarchical cross-entropy for few-shot. Eval suite already includes PlantNet, PlantVillage, Medicinal Leaf and PlantDoc.
→ Use: evidence that taxonomy-aware text embeddings generalize to unseen taxa. [Project](https://imageomics.github.io/bioclip), [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Stevens_BioCLIP_A_Vision_Foundation_Model_for_the_Tree_of_Life_CVPR_2024_paper.html)

**2. BioCLIP 2** (NeurIPS 2025 spotlight). ViT-L/14 trained on TreeOfLife-200M (214M images, 952K taxa from GBIF, EOL, BIOSCAN and FathomNet) with hierarchical contrastive learning. +18.1% species classification over BioCLIP; embeddings show emergent ecological structure. MIT license. Default model in `pybioclip`, which predicts at any rank from kingdom to species and has an embedding CLI.
→ Use: primary encoder, zero-shot baseline, and text tower for trait prototypes. [Project](https://imageomics.github.io/bioclip-2), [arXiv](https://arxiv.org/abs/2505.23883), [code](https://github.com/Imageomics/bioclip-2), [weights](https://huggingface.co/imageomics/bioclip-2), [pybioclip](https://github.com/imageomics/pybioclip)

**3. DINOv3** (Meta, Aug 2025). Self-supervised ViT-7B teacher trained on about 1.7B images, distilled into ViT-S/B/L/H+ and ConvNeXt T/S/B/L students. Model card: frozen features with simple heads, fine-tuning as a last resort. Hugging Face weights are gated (accept the license first).
→ Use: second encoder for fine visual detail (venation, margins). ConvNeXt-T for an on-device roadmap. [Blog](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/), [arXiv](https://arxiv.org/abs/2508.10104), [ViT-L weights](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)

**4. PlantCLEF 2024 / 2025** (LifeCLEF). Train on single-plant images, predict every species in high-resolution quadrat photos. The 2024 winner averaged 28.73 macro-F1 per plot. 2025 drew 38 teams and 659 methods, top scores around 0.36, and the organizers noted leaderboard overfitting from tuning too many hyperparameters. DS@GT (2nd in 2025) used the organizers' DINOv2 backbone with light heads and tiling.
→ Use: frozen backbone + light head is what strong teams do; the organizers' DINOv2 checkpoint is a ready plant encoder (trained on European flora, so check transfer to Indian species). [2025 overview](https://arxiv.org/abs/2509.17602), [DS@GT 2025](https://arxiv.org/abs/2507.06093) ([code](https://github.com/dsgt-arc/plantclef-2025)), [DS@GT 2024](https://arxiv.org/abs/2407.06298), [prototype-guided, 5th in 2025](https://arxiv.org/abs/2512.19957)

**5. Pl@ntNet-300K** (NeurIPS 2021 Datasets track). Built to evaluate set-valued classifiers because many plant photos are genuinely ambiguous, even to experts.
→ Use: main benchmark; justifies returning a set instead of one label. [GitHub](https://github.com/plantnet/PlantNet-300K)

**6. Flowers, leaves or both?** (Rzanny et al., Plant Methods 2019). 5 fixed views per plant, 101 species. Top-1: whole plant 77%, flower front 88%, all views fused 97%; flower front + side + leaf top gives 96%.
→ Use: multi-photo fusion and a "take a flower photo next" UX hint. [Paper](https://doi.org/10.1186/s13007-019-0462-4)

**7. VLM4Bio** (NeurIPS 2024 Datasets track). 12 VLMs on 469K QA pairs. GPT-4V got about 17% on open-ended species questions but 82.6% on bird multiple-choice; BioCLIP often doubled the best general VLMs on species classification.
→ Use: VLMs are poor identifiers but decent choosers over a short list, so use them only as rerankers. [arXiv](https://arxiv.org/abs/2408.16176)

**8. RAR: Retrieving And Ranking augmented MLLMs** (2024). A CLIP retriever pulls top-k candidates from an external memory; an MLLM ranks them. Gains on 5 fine-grained benchmarks and 11 few-shot datasets.
→ Use: template for the uncertainty-gated reranker. [arXiv](https://arxiv.org/abs/2403.13805)

**9. Classification by description** (Menon & Vondrick, ICLR 2023 oral). Score images against LLM-generated descriptors instead of bare class names: about 4-5% top-1 gain on ImageNet, built-in explanations, and recognition of concepts unseen in training.
→ Use: botanical trait descriptors (leaf margin, venation, phyllotaxy, flower structure) as text prototypes. Enables zero-image species and trait-level explanations. [arXiv](https://arxiv.org/abs/2210.07183)

**10. Geographic priors** (Mac Aodha et al., ICCV 2019) and **SINR** (Cole et al., ICML 2023). Multiply image probabilities by a "is this species found here, now?" prior: large gains on fine-grained datasets. SINR jointly estimates ranges for 47K species from presence-only data; pretrained models are public.
→ Use: EXIF GPS/date → range prior (GBIF counts or SINR). [Geo priors](https://arxiv.org/abs/1906.05272), [SINR](https://arxiv.org/abs/2306.02564) ([code](https://github.com/elijahcole/sinr))

**11. Conformal prediction for long-tailed plant ID** (Ding, Fermanian, Salmon 2025) and **multi-input conformal** (Fermanian et al. 2025). Standard conformal sets under-cover rare species; they propose prevalence-adjusted softmax and Fuzzy CP, tested on Pl@ntNet-300K. The multi-input paper aggregates per-photo conformal p-values so sets shrink as users add photos of the same plant while keeping class-conditional coverage.
→ Use: the "honest uncertainty" feature, with a guarantee. [Long-tail CP](https://arxiv.org/abs/2507.06867), [multi-input CP](https://arxiv.org/abs/2507.07150)

**12. SAM3 + DINOv3, training-free fine-grained segmentation** (2026). SAM3 gives class-agnostic masks from a coarse prompt; DINOv3 prototypes assign fine-grained labels. Demonstrated on mushrooms.
→ Use: the same pattern for "segment the leaf, classify the crop" with no training. [arXiv](https://arxiv.org/abs/2605.22492)

**13. Self-supervised plant representations** (2026 preprint). Gaussian blur, grayscale and solarization augmentations hurt plant SSL because they erase fine cues; pretraining on iNat21 Plantae beat ImageNet-1K.
→ Use: if you train any head or use test-time augmentation, avoid those augmentations. [Index page](https://www.newx.sg/paper/detail/acbcbf19-4513-11f1-b84c-00163e10baa7)

**14. Leafsnap** (Kumar et al., ECCV 2012). First mobile plant-ID app: rejects non-leaf images, segments the leaf, uses multi-scale contour curvature. Historical anchor for "leaf-only on plain background".
→ Use: background slide, and the "reject non-plant input" gate idea. [Paper](https://doi.org/10.1007/978-3-642-33709-3_36)

**15. Pl@ntNet API.** Free 500 identifications/day, 35,000+ species on the free tier, organ parameter, beta multi-species "survey" routes based on tiling.
→ Use: production baseline B0. [Pricing](https://my.plantnet.org/pricing), [survey routes](https://my.plantnet.org/doc/api/survey)

---

## 6. Recommended system (the novel angle)

Working name: **VanaLens** (rename freely).

```
photo(s) JPEG/PNG + optional EXIF GPS/date
  |
  1. Gate + organ router    BioCLIP 2 zero-shot: leaf / flower / fruit / bark / whole plant / not a plant
  2. Views                  full image + centre crop + object crop (optional SAM3 / GroundingDINO "leaf")
  3. Frozen encoders        BioCLIP 2 ViT-L/14 + DINOv3 ViT-L/16 (+ PlantCLEF DINOv2 ViT-B/14)
  4. Three scorers          (a) kNN over reference gallery, per organ   -> evidence images
                            (b) linear probe on concatenated features
                            (c) text prototypes: taxonomic name + LLM trait descriptors -> zero-image species
  5. Fuse + calibrate       weighted log-prob sum, temperature fitted on val
  6. Geo prior              x p(species | location) from GBIF facet counts or SINR
  7. Multi-photo fusion     sum log-probs across photos of the same plant
  8. Hierarchical backoff   species -> genus -> family when confidence is low
  9. Conformal set + reject 90% coverage set; "not in our flora" if max similarity < threshold
 10. Gated VLM reranker     only if set size > 1: Claude/Gemini sees query + top-k reference photos + traits
  |
output: species (or genus), calibrated set, 3 nearest reference photos, matched traits, "take a flower photo" hint
```

**Why it is new for a hackathon** (pitch bullets):

1. **Zero retraining to add a species**: drop 5 photos into the gallery, or give only a name + traits.
2. **Honest uncertainty**: a conformal set with a coverage guarantee and genus backoff instead of one overconfident label.
3. **Evidence, not vibes**: nearest reference photos + trait checklist with every answer. Useful for botanists and for medicinal-herb quality checks.
4. **VLM used only where it is strong**: reranking a short list on ambiguous cases (VLM4Bio's open-ended vs multiple-choice gap is the justification).
5. **Location-aware**: an India range prior removes look-alikes from other continents.
6. **Multi-organ**: a second photo measurably shrinks the set.

**India use cases for the pitch**: raw-herb adulteration screening (medicinal look-alikes), campus and forest biodiversity census, school field guides.

---

## 7. Implementation notes

### Environment

```bash
pip install pybioclip open_clip_torch transformers timm faiss-cpu scikit-learn gradio wandb pillow
# DINOv3 weights are gated: accept the license on its Hugging Face page, then `huggingface-cli login`
```

### B2 baseline in 5 lines

```python
from bioclip import TreeOfLifeClassifier, CustomLabelsClassifier, Rank
tol = TreeOfLifeClassifier()                         # BioCLIP 2 + TreeOfLife-200M embeddings
print(tol.predict("leaf.jpg", Rank.SPECIES)[:5])     # open-ended over ~950K taxa
closed = CustomLabelsClassifier(species_names)       # restrict to your label set
print(closed.predict("leaf.jpg")[:5])
```

### Embeddings (fp16 fits a 6 GB GPU)

```python
import torch, open_clip
from transformers import AutoImageProcessor, AutoModel
F = torch.nn.functional

bc, _, bc_pre = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
bc = bc.half().cuda().eval()
DID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
dproc = AutoImageProcessor.from_pretrained(DID)
dino = AutoModel.from_pretrained(DID, torch_dtype=torch.float16).cuda().eval()

@torch.no_grad()
def embed(pils):
    a = bc.encode_image(torch.stack([bc_pre(p) for p in pils]).half().cuda())
    px = dproc(images=pils, return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
    b = dino(pixel_values=px).pooler_output
    return torch.cat([F.normalize(a, dim=-1), F.normalize(b, dim=-1)], -1).float().cpu().numpy()
```

Cache every embedding to `.npy` keyed by image id. Cap the gallery at about 100 images per species: faster extraction and less head-class dominance.

### Scorers, fusion, conformal

```python
import numpy as np, faiss
from sklearn.linear_model import LogisticRegression

index = faiss.IndexFlatIP(Xg.shape[1]); index.add(Xg.astype("float32"))
def knn_proba(Xq, k=10, T=0.05):
    S, I = index.search(Xq.astype("float32"), k)
    P = np.zeros((len(Xq), n_cls))
    for i in range(len(Xq)):
        np.add.at(P[i], yg[I[i]], np.exp(S[i] / T))
    return P / P.sum(1, keepdims=True)

probe = LogisticRegression(max_iter=2000).fit(Xg, yg)   # or a 1-layer torch head on GPU if slow
def fuse(Xq, P_text, w=(0.4, 0.4, 0.2)):
    logp = (w[0] * np.log(probe.predict_proba(Xq) + 1e-9)
            + w[1] * np.log(knn_proba(Xq) + 1e-9)
            + w[2] * np.log(P_text + 1e-9))
    p = np.exp(logp - logp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)

# split conformal, 90% coverage, fitted on a held-out calibration split
P = fuse(Xcal, Pt_cal); s = 1 - P[np.arange(len(ycal)), ycal]
qhat = np.quantile(s, np.ceil((len(ycal) + 1) * 0.9) / len(ycal), method="higher")
pred_set = lambda p: np.flatnonzero(p >= 1 - qhat)
```

`predict_proba` columns follow `probe.classes_`, so make sure every species appears in the gallery. For the long tail, swap the score for the prevalence-adjusted version from the long-tail CP paper.

### Text prototypes (zero-image species + explanations)

1. Ask Claude once per species (batch): *"List 6 short visual traits to identify {species} from a photo of its {organ}: leaf shape, margin, venation, arrangement, flower colour/structure, bark."*
2. Encode `"a photo of {species}, which has {trait}"` with the BioCLIP 2 text tower. Species score = mean cosine over its traits; softmax → `P_text`.
3. Keep the per-trait cosines: the top-matching traits become the explanation in the UI.

### Geo prior

- GBIF occurrence search takes `geoDistance=lat,lng,50km` together with `facet=speciesKey&facetLimit=...&limit=0`, which returns per-species counts around a point in one call (check exact parameter format in the GBIF API reference). Cache by grid cell.
- `prior = (count + 1) / sum(count + 1)`, then `p_geo ∝ p * prior**beta`. Tune `beta` on val, start at 0.5.
- Pl@ntNet-300K and DIMPSAR may lack coordinates: evaluate the prior on the campus set and on an iNat-derived India set built from the AWS open data.

### Hierarchical backoff

Sum species probabilities into genus and family. Return the deepest rank with probability of at least 0.6 (tune on val), e.g. "Ficus (94%); species uncertain between F. religiosa and F. benghalensis".

### Gated VLM reranker

Run only when the conformal set has more than 1 species. Send the query photo, 2 reference photos per candidate (top-5) and each candidate's traits. Ask for a pick from the list plus 2 visible traits that decide it. Log accuracy with and without it on the ambiguous subset; that delta is a pitch slide.

---

## 8. 24-hour schedule

Assumes 3 people: **ML**, **Data/Eval**, **App/Pitch**. Solo: drop the reranker and the geo prior, keep everything else.

| Hours | ML | Data / Eval | App / Pitch | Checkpoint |
|---|---|---|---|---|
| 0-1 | Env, HF / Kaggle / W&B tokens, MCPs. Accept DINOv3 license | Start downloads: DIMPSAR first, Pl@ntNet-300K straight into Kaggle. Pl@ntNet API key | Repo, Gradio skeleton | Data flowing |
| 1-3 | B2 BioCLIP 2 zero-shot on DIMPSAR + 2K Pl@ntNet-300K test subset | B1 repo weights; B0 Pl@ntNet API on 400 images | Upload → BioCLIP 2 top-5 in UI | **Baseline table v0** |
| 3-6 | Extract embeddings (both encoders) for capped galleries + test sets on Kaggle | Eval harness (all metrics, W&B). **Shoot campus photos now in daylight** (15-20 species x 5) | Evidence panel (nearest images) | Embeddings cached |
| 6-9 | kNN + probe + fusion weights on val, temperature scaling | Few-shot curve (1 / 5 / 10 shots) | Wire fused model into UI | **MVP gate: working fused demo** |
| 9-12 | LLM trait descriptors, text prototypes, hierarchical backoff | Conformal sets + open-set threshold. Ablation v1 | Show set + genus backoff in UI | Ablation v1 |
| 12-14 | Multi-photo fusion. Geo prior from GBIF facets | Campus / India gallery from iNat AWS open data | "Add a species" few-shot uploader | |
| 14-16 | Gated VLM reranker on the ambiguous subset (cost, latency) | Final runs on all test sets incl. campus | Deploy to HF Space | Ablation v2 |
| 16-19 | Fix failures from error analysis | Results tables, charts | Polish UI, mobile check | **Feature freeze at 19** |
| 19-22 | Support pitch, record demo video | Re-check every number against logs | 8-slide deck | Deck v1 |
| 22-24 | Buffer | Buffer | Rehearse twice | Submit |

**Cut ladder if behind**: reranker → geo prior → DINOv3 (BioCLIP 2 only) → text prototypes. Never cut: baseline table, conformal set, evidence panel, campus test.

---

## 9. Tooling: MCPs, skills, compute

### MCP servers for Claude Code

| Tool | What it gives you | Setup |
|---|---|---|
| **Hugging Face MCP** (official) | Search models, datasets, papers, Spaces; HF docs | `claude mcp add hf-mcp-server -t http https://huggingface.co/mcp?login` |
| **Hugging Face Skills** | Agent skills incl. `hugging-face-vision-trainer` (image classification) and `hugging-face-jobs`. HF Jobs need a paid HF plan | `/plugin marketplace add huggingface/skills` then `/plugin install hugging-face-vision-trainer@huggingface-skills` |
| **Weights & Biases MCP** (official) | Query runs and sweeps, compare ablations, write W&B reports | `claude mcp add --transport http wandb https://mcp.withwandb.com/mcp --header "Authorization: Bearer $WANDB_API_KEY"` |
| **Kaggle MCP** (official, remote) | Datasets, notebooks, models, competitions | Remote URL `https://www.kaggle.com/mcp` (OAuth). Community alternative: `claude mcp add kaggle -- uvx kaggle-mcp-server` |
| alphaXiv | Full-text arXiv search and Q&A over papers | claude.ai connector |
| Context7 | Current docs for open_clip, transformers, faiss | `claude mcp add --transport http context7 https://mcp.context7.com/mcp` |
| GitHub MCP | Repo, issues, PRs during the sprint | Optional |

### Claude skills / plugins

| Skill | Use |
|---|---|
| `frontend-design` | Demo UI that doesn't look generic |
| Slides artifact or `pptx` | Final pitch deck |
| `dataviz` | Ablation and few-shot charts |
| `skill-creator` | Package the eval harness as a reusable skill ("run plant eval on folder X") |
| `deep-research` | Follow-up literature passes |
| Probabl Skills (plugin directory) | Build / evaluate / smoke-test ML pipeline skills |
| Pitch Deck Advisor (plugin directory) | Deck review against YC / Sequoia guidance |

### Compute

| Resource | Use |
|---|---|
| Local RTX 4050 / 3050, 6 GB | fp16 inference of both ViT-L encoders, probes, demo |
| MacBook M3 Pro, 18 GB | Backup inference (MPS), deck work |
| Kaggle notebooks | Bulk embedding extraction: about 30 GPU-h/week, 2x T4 or P100, 12 h sessions |
| Colab free | Overflow; T4 not guaranteed |
| HF Spaces (ZeroGPU) | Public demo; small daily GPU quota on free accounts |

---

## 10. Pitch outline (8 slides)

1. **Problem**: plant-ID apps give one overconfident answer; medicinal look-alikes get mixed up; adding a local species means retraining.
2. **Why current approaches fall short**: fine-tuned models are closed-set; general VLMs score about 17% open-ended; no calibrated uncertainty.
3. **Our system**: the Section 6 diagram.
4. **Live demo**: campus leaf → set + evidence + traits; add a second photo → set shrinks; add a new species from 5 photos.
5. **Results**: vs Pl@ntNet API, BioCLIP 2 zero-shot, VLM, Pl@ntNet-300K CNN. Top-1, macro top-1, set size at 90% coverage.
6. **Ablation**: + kNN, + text traits, + geo prior, + multi-photo, + reranker.
7. **Clean test**: campus photos nobody's model has seen.
8. **Impact + roadmap**: herb QC, biodiversity census, distilled ConvNeXt for offline mobile.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Pl@ntNet-300K is 31.7 GB | Pull straight into Kaggle. If slow, evaluate on a 5K-image test subset with a capped gallery |
| DINOv3 access is gated | Request at hour 0. Fallback: DINOv2 or the PlantCLEF DINOv2 checkpoint |
| BioCLIP 2 / Pl@ntNet API may have seen test images | Report the campus set separately and say it on the slide |
| Pl@ntNet API cap (500/day) | Sample 400 images, run once, cache |
| VLM cost / latency | Gate on set size > 1, cap at 300 calls |
| Demo-day lighting or Wi-Fi | Pre-recorded video + local Gradio build |

---

## Sources

**Datasets**
- Pl@ntNet-300K: [GitHub](https://github.com/plantnet/PlantNet-300K) · [Zenodo](https://zenodo.org/records/5645731) · [NeurIPS paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/7e7757b1e12abcb736ab9a754ffb617a-Paper-round2.pdf)
- PlantCLEF 2025: [challenge page](https://www.imageclef.org/PlantCLEF2025) · [overview](https://arxiv.org/abs/2509.17602) · [pretrained DINOv2 models](https://doi.org/10.5281/zenodo.10848263)
- iNaturalist 2021: [competition repo](https://github.com/visipedia/inat_comp/tree/master/2021)
- iNaturalist Open Data: [AWS registry](https://registry.opendata.aws/inaturalist-open-data) · [docs](https://github.com/inaturalist/inaturalist-open-data)
- TreeOfLife-200M: [Hugging Face](https://huggingface.co/datasets/imageomics/TreeOfLife-200M)
- DIMPSAR: [Mendeley](https://data.mendeley.com/datasets/748f8jkphb) · [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10375553/)
- MED117: [Mendeley](https://data.mendeley.com/datasets/dtvbwrhznz/2)
- HerbLife: [Mendeley](https://data.mendeley.com/datasets/v8n2twxxdb)
- Medicinal Leaf Dataset: [Mendeley](https://data.mendeley.com/datasets/nnytj2v3n5)
- SIMPD: [Data in Brief](https://www.sciencedirect.com/science/article/pii/S2352340925003907)
- LeafSnap: [dataset](http://leafsnap.com/dataset/) · [ECCV 2012 paper](https://doi.org/10.1007/978-3-642-33709-3_36)
- Herbarium 2021: [Frontiers](https://www.frontiersin.org/articles/10.3389/fpls.2021.787127) · NAFlora-1M / Herbarium 2022: [GitHub](https://github.com/dpl10/NAFlora-1M) · [Kaggle](https://www.kaggle.com/c/herbarium-2022-fgvc9) · [competition repo](https://github.com/visipedia/herbarium_comp)
- Sulc & Matas 2014 (leaf texture, Flavia): [paper](https://doi.org/10.1007/978-3-319-16220-1_14)

**Models and methods**
- BioCLIP: [project](https://imageomics.github.io/bioclip) · [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Stevens_BioCLIP_A_Vision_Foundation_Model_for_the_Tree_of_Life_CVPR_2024_paper.html)
- BioCLIP 2: [project](https://imageomics.github.io/bioclip-2) · [arXiv](https://arxiv.org/abs/2505.23883) · [code](https://github.com/Imageomics/bioclip-2) · [weights](https://huggingface.co/imageomics/bioclip-2) · [pybioclip](https://github.com/imageomics/pybioclip)
- DINOv3: [Meta blog](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/) · [arXiv](https://arxiv.org/abs/2508.10104) · [ViT-L/16 weights](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- PlantCLEF solutions: [DS@GT 2025](https://arxiv.org/abs/2507.06093) · [code](https://github.com/dsgt-arc/plantclef-2025) · [DS@GT 2024](https://arxiv.org/abs/2407.06298) · [prototype-guided 2025](https://arxiv.org/abs/2512.19957)
- Rzanny et al. 2019, multi-organ: [Plant Methods](https://doi.org/10.1186/s13007-019-0462-4)
- VLM4Bio: [arXiv](https://arxiv.org/abs/2408.16176)
- RAR: [arXiv](https://arxiv.org/abs/2403.13805)
- Classification by description: [arXiv](https://arxiv.org/abs/2210.07183)
- Geo priors: [arXiv](https://arxiv.org/abs/1906.05272) · SINR: [arXiv](https://arxiv.org/abs/2306.02564) · [code](https://github.com/elijahcole/sinr)
- Conformal, long-tailed: [arXiv](https://arxiv.org/abs/2507.06867) · multi-input: [arXiv](https://arxiv.org/abs/2507.07150)
- SAM3 + DINOv3 fine-grained segmentation: [arXiv](https://arxiv.org/abs/2605.22492)
- Self-supervised plant representations: [index page](https://www.newx.sg/paper/detail/acbcbf19-4513-11f1-b84c-00163e10baa7)
- DINOv3 iNat2021 linear vs fine-tune numbers: [aggregator](https://www.wizwand.com/sota/image-classification-on-inaturalist-2021)

**APIs and tooling**
- Pl@ntNet API: [pricing](https://my.plantnet.org/pricing) · [terms](https://my.plantnet.org/terms_of_use) · [survey routes](https://my.plantnet.org/doc/api/survey)
- GBIF API: [reference](https://techdocs.gbif.org/en/openapi/) · [rgbif counts and facets guide](https://docs.ropensci.org/rgbif/articles/occ_counts.html)
- Hugging Face MCP: [GitHub](https://github.com/huggingface/hf-mcp-server) · HF Skills: [docs](https://huggingface.co/docs/hub/agents-skills) · [blog](https://huggingface.co/blog/hf-skills-training)
- W&B MCP: [docs](https://docs.wandb.ai/platform/mcp-server)
- Kaggle MCP (community, documents the official remote server): [GitHub](https://github.com/Galaxy-Dawn/kaggle-mcp) · [PyPI](https://pypi.org/project/kaggle-mcp-server/)
- Free GPU quotas: [Kaggle / Colab overview, Sep 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
