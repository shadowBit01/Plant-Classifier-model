"""Small dataset checks for the L7-d review, from vl600 meta.csv (outside the repo).

Writes pitch/report/review_stats.json:
- share of vl600 test photos/observations taken in India (geographic bias),
- test observations and photos per species (min / median / max),
- a contamination proxy: test observations whose rounded coordinates (4 decimals, about 11 m) match a gallery or
  calibration observation of the same species, i.e. probably the same individual plant photographed again
  after the cutoff (date cut does not separate those).

Run: python pitch/report/review_stats.py
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vanalens.config import DATA_DIR  # noqa: E402

CUT = "2025-07-01"
OUT = Path(__file__).with_name("review_stats.json")


def main() -> None:
    rows = list(csv.DictReader(open(DATA_DIR / "vl600" / "meta.csv", encoding="utf-8")))
    test, pre = [], []
    for r in rows:
        sp = Path(r["image"]).parts[0] if "/" in r["image"] or "\\" in r["image"] else r.get("species", "")
        r["_sp"] = sp
        (test if (r["observed_on"] or "") >= CUT else pre).append(r)
    obs_test = {r["obs_id"]: r for r in test}
    india = lambda r: str(r.get("in_india", "")).lower() in ("1", "true", "yes")
    per_sp_obs, per_sp_ph = defaultdict(set), defaultdict(int)
    for r in test:
        per_sp_obs[r["_sp"]].add(r["obs_id"])
        per_sp_ph[r["_sp"]] += 1

    def key(r):
        try:
            return (r["_sp"], round(float(r["lat"]), 4), round(float(r["lon"]), 4))
        except (TypeError, ValueError):
            return None

    pre_keys = {k for k in map(key, pre) if k}
    with_gps = [r for r in obs_test.values() if key(r)]
    same_spot = [r for r in with_gps if key(r) in pre_keys]
    no = [len(v) for v in per_sp_obs.values()]
    npht = list(per_sp_ph.values())
    out = {
        "n_test_photos": len(test), "n_test_observations": len(obs_test),
        "india_share_test_photos": round(sum(map(india, test)) / max(len(test), 1), 4),
        "india_share_test_observations": round(sum(map(india, obs_test.values())) / max(len(obs_test), 1), 4),
        "species_with_test": len(per_sp_obs),
        "test_obs_per_species": {"min": min(no), "median": statistics.median(no), "max": max(no)},
        "test_photos_per_species": {"min": min(npht), "median": statistics.median(npht), "max": max(npht)},
        "test_obs_with_gps": len(with_gps),
        "test_obs_same_spot_as_gallery_or_cal": len(same_spot),
        "same_spot_share": round(len(same_spot) / max(len(with_gps), 1), 4),
        "same_spot_rule": "same species and lat/lon equal after rounding to 4 decimals (about 11 m)",
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
