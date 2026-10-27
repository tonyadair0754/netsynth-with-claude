"""
check_group_sizes_vs_inputs.py -- INDEPENDENT check of the release's school/daycare group sizes.

The release also ships its INPUT site lists (edu_site_by_state/<st>_edu_sites.zip -> <st>_school_id.gpkg and
<st>_daycare_id.gpkg), which carry an ENROLLMENT column from the source site database. That is real-looking,
independent information about how many children each site serves, so we can compare it with how many children the
generator actually assigned to each site.

Usage (from the repo root): python release_tools/check_group_sizes_vs_inputs.py <release_dir (has <st>_population.gpkg)> <edu_dir (has *_id.gpkg)> [st]
Writes results/group_sizes/group_size_vs_inputs_<st>.csv
CAUTION: the input databases are from 2015 and list licensed centres/schools; enrollment is not a full census of
children, so large gaps are a reason to look closer, not proof that the generator is "wrong".
"""
import sys
from pathlib import Path

import pandas as pd
import pyogrio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # make the repo root importable
import paths  # noqa: E402

release, edu = Path(sys.argv[1]), Path(sys.argv[2])
st = sys.argv[3] if len(sys.argv) > 3 else "ut"
pop = pyogrio.read_dataframe(release / f"{st}_population.gpkg", columns=["wp", "age"], read_geometry=False)
rows = []
for kind in ("daycare", "school"):
    e = pyogrio.read_dataframe(edu / f"{st}_{kind}_id.gpkg", columns=["eduID", "ENROLLMENT"], read_geometry=False)
    e["ENROLLMENT"] = pd.to_numeric(e.ENROLLMENT, errors="coerce")
    s = pop[pop.wp.fillna("").str.contains("d" if kind == "daycare" else "s", regex=False) & ~pop.wp.fillna("").str.contains("w")]
    s = s.groupby("wp").size().rename("assigned")
    d = e.rename(columns={"eduID": "wp"}).set_index("wp").join(s, how="left").fillna({"assigned": 0})
    ok = d[d.ENROLLMENT > 0]
    ratio = ok.assigned / ok.ENROLLMENT
    rows.append({"layer": kind, "sites_in_input": len(d), "input_total_enrollment": d.ENROLLMENT.sum(),
                 "release_total_assigned": d.assigned.sum(), "assigned_per_enrolled_total": d.assigned.sum() / d.ENROLLMENT.sum(),
                 "input_median": ok.ENROLLMENT.median(), "release_median": d.assigned[d.assigned > 0].median(),
                 "input_p99": ok.ENROLLMENT.quantile(.99), "release_p99": d.assigned[d.assigned > 0].quantile(.99),
                 "input_max": ok.ENROLLMENT.max(), "release_max": d.assigned.max(),
                 "spearman_enrollment_vs_assigned": ok[["ENROLLMENT", "assigned"]].corr(method="spearman").iloc[0, 1],
                 "median_assigned_over_enrollment": ratio.median(), "p90_assigned_over_enrollment": ratio.quantile(.9)})
out = pd.DataFrame(rows)
(paths.RESULTS / "group_sizes").mkdir(parents=True, exist_ok=True)
out.to_csv(paths.RESULTS / "group_sizes" / f"group_size_vs_inputs_{st}.csv", index=False)
pd.set_option("display.width", 220); print(out.round(3).T.to_string())
