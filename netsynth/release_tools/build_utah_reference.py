"""
build_utah_reference.py -- distil the 2 GB Utah release into a SMALL reference folder (a few MB) so the
generator experiments can run without the raw data.

Writes data/utah_reference/:
    household_sample.csv.gz   50,000 whole households (hh, age): template for household size + age composition
    size_hist_<layer>.csv     empirical group-size histograms (size, count) for work / school / daycare
    (plus copies of the release_features_<st>/ tables used as the comparison reference)
NOTE: these are properties of a synthetic release, not of real people.
Usage (from the repo root): python release_tools/build_utah_reference.py /path/to/ut ut   (run analyze_release.py first)
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # make the repo root importable
import paths  # noqa: E402

folder, st = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "ut"
feat = paths.RESULTS / f"release_features_{st}"
out = paths.UTAH_REFERENCE; out.mkdir(parents=True, exist_ok=True)
pop = pyogrio.read_dataframe(folder / f"{st}_population.gpkg", columns=["id", "age", "hhold", "htype", "wp"], read_geometry=False)
pop = pop[pop.id.notna() & (pop.htype != 11)]                       # drop placeholder row and group quarters
hh = pop.groupby("hhold").age.apply(list)
rng = np.random.default_rng(0)
pick = rng.choice(len(hh), 50_000, replace=False)
rows = [(i, a) for i, k in enumerate(pick) for a in hh.iloc[k]]
pd.DataFrame(rows, columns=["hh", "age"]).to_csv(out / "household_sample.csv.gz", index=False)
for k in ("work", "school", "daycare"):
    s = np.load(feat / f"group_sizes_{k}.npy")
    v, c = np.unique(s, return_counts=True)
    pd.DataFrame({"size": v, "count": c}).to_csv(out / f"size_hist_{k}.csv", index=False)
for f in feat.glob("*.csv"):
    shutil.copy(f, out / f.name)
shutil.copy(feat / "release_summary.json", out / "release_summary.json")
print("wrote", out.resolve(), sorted(p.name for p in out.iterdir()))
