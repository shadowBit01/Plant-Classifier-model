"""Pick concrete demo photos for pitch/demo.md from cached vl600 test embeddings (CPU, no GPU, no network).

Uses the same VanaLens.scores / combine path as the app on the cached embeddings, with the GBIF prior read from
cached cells only (no network). Writes pitch/demo_picks.json with:
- known: Indian test photos named correctly at species rank with high confidence (1 photo, no GPS),
- geo: photos whose top-1 is wrong without GPS and right with GPS (location prior effect),
- multi: Indian observations with 3 photos where the best single photo is wrong or unsure and the 3-photo answer is right,
- unknown: Vanda tessellata (an orchid in inat_india but not among the 607 vl600 classes): per test photo, whether the
  open-set gate rejects it before it is added, and whether it is named after appending 5 of its inat_india gallery
  photos (cached embeddings, same encoders; add_species without re-encoding, image-centroid text prototype).

Run: python pitch/demo_pick.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vanalens import config, geo  # noqa: E402
from vanalens.contracts import EmbeddingSet  # noqa: E402
from vanalens.pipeline import VanaLens  # noqa: E402

OUT = Path(__file__).with_name("demo_picks.json")


def cached_counts(lat: float, lon: float, radius_km: int = 50, kingdom_key: int = 6, cache_dir=None):
    """GBIF counts from the local cell cache only; empty dict when the cell is not cached (no network)."""
    cell_lat, cell_lon = round(lat * 2) / 2, round(lon * 2) / 2
    f = config.CACHE_DIR / "gbif" / f"{cell_lat}_{cell_lon}_{radius_km}_{kingdom_key}.json"
    if not f.exists():
        return {}
    return {int(k): v for k, v in json.loads(f.read_text()).items()}


def unknown_demo(lens: VanaLens, names: list[str], species: str = "Vanda tessellata", shots: int = 5) -> dict:
    """Reject-then-add check for one species missing from vl600, on cached inat_india embeddings."""
    from vanalens import conformal, hierarchy
    from vanalens.contracts import Taxon
    from vanalens.contracts import load_taxa
    tin = load_taxa(config.DATA_DIR / "inat_india" / "labels.json")
    c = next(t.idx for t in tin if t.species == species)
    ga = EmbeddingSet.load(config.CACHE_DIR / "emb" / "inat_india_gallery.npz")
    te = EmbeddingSet.load(config.CACHE_DIR / "emb" / "inat_india_test.npz")
    paths = {}
    for split in ("gallery", "test"):
        for line in open(config.CACHE_DIR / "manifests" / f"inat_india_{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            paths[r["image_id"]] = r["path"]
    gi = [i for i in range(len(ga.ids)) if int(ga.y[i]) == c][:shots]
    ti = [i for i in range(len(te.ids)) if int(te.y[i]) == c]

    def judge(X):
        p = lens.combine(lens.scores(X))
        sim = float(lens.knn.max_similarity(X).max())
        unk = bool(conformal.is_unknown(np.array([sim]), lens.thr)[0])
        rank, label, conf = hierarchy.backoff(p, lens.taxa, 0.6)
        return {"unknown": unk, "sim": round(sim, 3), "rank": rank, "label": label, "conf": round(float(conf), 3),
                "top1": lens.taxa[int(p.argmax())].species}
    before = [judge(te.X[i:i + 1]) for i in ti]
    t0 = next(t for t in tin if t.idx == c)
    idx = len(lens.taxa)
    from dataclasses import replace
    lens.taxa.append(replace(t0, idx=idx))
    lens.knn.n_classes = len(lens.taxa)
    ids = [f"added/{species}/{k}" for k in range(len(gi))]
    lens.knn.add(EmbeddingSet(ids, ga.X[gi], np.full(len(gi), idx, dtype=np.int64)))
    if lens.text is not None:
        from vanalens.pipeline import _l2, BIOCLIP_DIM
        lens.text.protos.append(_l2(_l2(ga.X[gi][:, :BIOCLIP_DIM]).mean(0)))
        lens.text.trait_strings.append([f"a photo of {species}"])
    after = [judge(te.X[i:i + 1]) for i in ti]
    return {"species": species, "thr": round(float(lens.thr), 3), "shots": len(gi),
            "add_paths": [paths.get(ga.ids[i]) for i in gi],
            "test": [{"path": paths.get(te.ids[i]), "before": b, "after": a} for i, b, a in zip(ti, before, after)],
            "rejected_before": sum(b["unknown"] for b in before), "named_after": sum(a["top1"] == species for a in after),
            "n_test": len(ti)}


def main() -> None:
    geo.gbif_species_counts = cached_counts
    lens = VanaLens.from_cache("vl600")
    te = EmbeddingSet.load(config.CACHE_DIR / "emb" / "vl600_test.npz")
    man = {}
    for line in open(config.CACHE_DIR / "manifests" / "vl600_test.jsonl", encoding="utf-8"):
        r = json.loads(line)
        man[r["image_id"]] = r
    meta = {}
    import csv
    for r in csv.DictReader(open(config.DATA_DIR / "vl600" / "meta.csv", encoding="utf-8")):
        meta[r["image"].rsplit(".", 1)[0]] = r
    P = lens.scores(te.X)
    names = [t.species for t in lens.taxa]
    known, geo_flip, groups = [], [], defaultdict(list)
    for i, iid in enumerate(te.ids):
        m = man.get(iid) or {}
        rel = iid.split("/", 1)[1] if "/" in iid else iid
        mt = meta.get(rel, {})
        india = str(mt.get("in_india", "")).lower() == "true"
        lic = mt.get("license", "")
        y = int(te.y[i])
        groups[m.get("group", iid)].append(i)
        p0 = P[i]
        top0 = int(p0.argmax())
        rec = {"image_id": iid, "path": m.get("path"), "species": names[y], "lat": m.get("lat"), "lon": m.get("lon"),
               "india": india, "license": lic}
        if india and top0 == y and p0[top0] > 0.97 and lic.lower() in ("cc-by", "cc0"):
            known.append(dict(rec, conf=round(float(p0[top0]), 4)))
        if m.get("lat") is not None and india:
            pg = lens.combine(P[i:i + 1], m["lat"], m["lon"])
            tg = int(pg.argmax())
            if top0 != y and tg == y:
                geo_flip.append(dict(rec, no_gps=names[top0], no_gps_conf=round(float(p0[top0]), 3),
                                     gps_conf=round(float(pg[tg]), 3)))
    multi = []
    for g, idx in groups.items():
        if len(idx) != 3:
            continue
        y = int(te.y[idx[0]])
        m0 = man.get(te.ids[idx[0]]) or {}
        rel = te.ids[idx[0]].split("/", 1)[1]
        if str(meta.get(rel, {}).get("in_india", "")).lower() != "true":
            continue
        singles = [(int(P[i].argmax()), float(P[i].max())) for i in idx]
        p3 = lens.combine(P[idx])
        if int(p3.argmax()) == y and any(s[0] != y for s in singles):
            multi.append({"group": g, "species": names[y], "paths": [man[te.ids[i]]["path"] for i in idx],
                          "singles": [[names[c], round(p, 3)] for c, p in singles],
                          "three_conf": round(float(p3.max()), 3)})
    out_unknown = unknown_demo(lens, names)
    known.sort(key=lambda r: -r["conf"])
    out = {"known": known[:8], "geo": geo_flip[:8], "multi": multi[:8], "unknown": out_unknown,
           "counts": {"known": len(known), "geo": len(geo_flip), "multi": len(multi)}}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["counts"]))


if __name__ == "__main__":
    main()
