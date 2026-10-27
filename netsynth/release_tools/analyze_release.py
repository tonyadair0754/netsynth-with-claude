"""
analyze_release.py -- extract a feature profile from one state of the geo-synthetic-pop-usa release.

Input folder layout (as published on OSF, e.g. ut.zip):
    <dir>/<st>_population.gpkg            people: id, age, gender, hhold, htype, wp, urban, long, lat
    <dir>/<st>_workplace.gpkg             (optional) workplace points: wp, long, lat
    <dir>/social_networks/<st>_{household,school,daycare}_network.csv   adjacency lists (person, nbr, nbr, ...)
    <dir>/social_networks/<st>_work_network.csv                         EDGE LIST (index, source, target)

What it measures (all reproducible; sampling uses a fixed seed):
    * group-size distribution per layer (household / work / school / daycare)
    * per-layer and merged degree, clustering, connected components, path length (sampled)
    * age-group mixing matrix per layer (the same kind of object as the SES mixing matrix)
    * check that each layer's edges follow the repository rule (clique if size <= 5, else NWS k=4,p=0.3)
    * synthetic-workplace count versus number of workers

IMPORTANT: this is a release of a *model*, not observed ties. These numbers describe what the generator's
assumptions produce, not how real people are connected.

Usage (from the repo root): python release_tools/analyze_release.py /path/to/ut [st]    ->  results/release_features_<st>/
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # make the repo root importable
import paths   # noqa: E402
from layered import AGE_BINS, AGE_NAMES, age_mixing, clustering_sample, gini   # shared with the generator

SEED = 0




def load_adjacency_list(path, idx, chunk=400_000):
    """Headerless adjacency list -> symmetric 0/1 sparse matrix (ids mapped through `idx`)."""
    R, C = [], []

    def flush(buf):
        lens, flat = [], []
        for line in buf:
            fs = line.rstrip("\r\n").split(",")
            lens.append(len(fs) - 1); flat.extend(fs)
        ids = idx.get_indexer(flat)
        lens = np.array(lens)
        starts = np.concatenate([[0], np.cumsum(lens + 1)[:-1]])
        mask = np.ones(len(flat), bool); mask[starts] = False
        return np.repeat(ids[starts], lens), ids[mask]

    buf = []
    with open(path) as f:
        for line in f:
            if line.strip():
                buf.append(line)
            if len(buf) >= chunk:
                r, c = flush(buf); R.append(r); C.append(c); buf = []
    if buf:
        r, c = flush(buf); R.append(r); C.append(c)
    r, c = np.concatenate(R), np.concatenate(C)
    ok = (r >= 0) & (c >= 0)
    n = len(idx)
    A = sp.csr_matrix((np.ones(ok.sum(), np.int8), (r[ok], c[ok])), shape=(n, n))
    A = A.maximum(A.T); A.data[:] = 1
    return A


def load_edge_list(path, idx):
    w = pd.read_csv(path, index_col=0)
    r, c = idx.get_indexer(w.source), idx.get_indexer(w.target)
    ok = (r >= 0) & (c >= 0); n = len(idx)
    A = sp.csr_matrix((np.ones(ok.sum(), np.int8), (r[ok], c[ok])), shape=(n, n))
    A = A.maximum(A.T); A.data[:] = 1
    return A






def main(folder, st):
    folder = Path(folder); out = paths.RESULTS / f"release_features_{st}"; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pop = pyogrio.read_dataframe(folder / f"{st}_population.gpkg", read_geometry=False)
    pop = pop[pop.id.notna()].reset_index(drop=True)          # the release has one all-NaN placeholder row
    N = len(pop); idx = pd.Index(pop.id.values)
    kind = np.where(pop.wp.isna(), "none", np.where(pop.wp.str.contains("w"), "work",
                    np.where(pop.wp.str.contains("s"), "school", "daycare")))
    pop["kind"] = kind
    age = pop.age.values; ag = np.digitize(age, AGE_BINS)
    rep = {"people": N, "people_by_daytime_kind": pd.Series(kind).value_counts().to_dict(),
           "max_age": float(age.max()), "n_age_ge_90": int((age >= 90).sum())}

    # ---------------- group sizes ----------------
    sizes = {}
    hs = pop.groupby("hhold").size(); gq = set(pop.loc[pop.htype == 11, "hhold"])
    sizes["household"] = hs[~hs.index.isin(gq)]; sizes["group_quarters"] = hs[hs.index.isin(gq)]
    for k in ("work", "school", "daycare"):
        sizes[k] = pop[pop.kind == k].groupby("wp").size()
    rows = []
    for k, s in sizes.items():
        rows.append({"layer": k, "groups": len(s), "people": int(s.sum()), "mean": s.mean(), "median": s.median(),
                     "p90": s.quantile(.9), "p99": s.quantile(.99), "max": int(s.max()), "share_size_1": float((s == 1).mean()),
                     "share_size_le5": float((s <= 5).mean())})
    gs = pd.DataFrame(rows); gs.to_csv(out / "group_size_summary.csv", index=False)
    sizes["household"].value_counts(normalize=True).sort_index().rename("share").to_csv(out / "household_size_pmf.csv")
    for k in ("work", "school", "daycare"):
        np.save(out / f"group_sizes_{k}.npy", sizes[k].values)
    adults = pop[(pop.age >= 18)]
    rep["employed_share_of_adults"] = float((adults.kind == "work").mean())
    print(gs.round(3).to_string(index=False))

    # ---------------- layers ----------------
    L = {}
    sd = folder / "social_networks"
    for layer in ("household", "school", "daycare"):
        L[layer] = load_adjacency_list(sd / f"{st}_{layer}_network.csv", idx)
    L["work"] = load_edge_list(sd / f"{st}_work_network.csv", idx)
    M = sum(m.astype(np.int8) for m in L.values()); L["merged"] = (M > 0).astype(np.int8).tocsr()
    del M; gc.collect()
    S = rng.choice(N, min(100_000, N), replace=False)
    feats = {}
    for k, A in L.items():
        d = np.asarray(A.sum(axis=1)).ravel()
        c, kk = clustering_sample(A, S)
        P, assort = age_mixing(A, ag)
        pd.DataFrame(P, index=AGE_NAMES, columns=AGE_NAMES).to_csv(out / f"age_mixing_{k}.csv")
        feats[k] = {"edges": int(A.nnz // 2), "nodes_with_edges": int((d > 0).sum()), "mean_degree_all": d.mean(),
                    "mean_degree_active": d[d > 0].mean(), "max_degree": int(d.max()), "degree_gini_all": gini(d),
                    "avg_clustering_all_nodes": float(c.mean()),
                    "avg_clustering_nodes_deg_ge2": float(c[kk > 1].mean()) if (kk > 1).any() else np.nan,
                    "age_assortativity": assort}
    d = np.asarray(L["merged"].sum(axis=1)).ravel()
    np.save(out / "merged_degree_hist.npy", np.bincount(d.astype(int)) / N)
    nc, lab = connected_components(L["merged"], directed=False); cs = np.bincount(lab)
    gi = np.where(lab == cs.argmax())[0]
    src = rng.choice(gi, 8, replace=False)
    D = dijkstra(L["merged"], directed=False, unweighted=True, indices=src)[:, gi]
    feats["merged"].update({"components": int(nc), "giant_share": cs.max() / N, "isolates": int((cs == 1).sum()),
                            "avg_path_giant_sampled8": float(D[D > 0].mean()), "max_distance_seen": int(D[np.isfinite(D)].max())})
    pd.DataFrame(feats).T.to_csv(out / "layer_features.csv")
    print(pd.DataFrame(feats).T.round(4).to_string())

    # ---------------- does each layer follow the repo's rule? ----------------
    chk = []
    for layer, key, A in (("household", "hhold", L["household"]), ("school", "wp", L["school"]),
                          ("daycare", "wp", L["daycare"]), ("work", "wp", L["work"])):
        sub = pop[pop.kind == layer] if layer != "household" else pop[pop.htype != 11]
        g = sub.groupby(key).size(); sub = sub.assign(gs=sub[key].map(g).values)
        dg = np.asarray(A.sum(axis=1)).ravel()[sub.index.values]
        sub = sub.assign(deg=dg)
        small = sub[sub.gs <= 5]; big = sub[sub.gs > 5]
        chk.append({"layer": layer, "small_groups_deg_equals_size_minus_1": float((small.deg == small.gs - 1).mean()) if len(small) else np.nan,
                    "big_group_mean_degree": float(big.deg.mean()) if len(big) else np.nan,
                    "big_group_min_degree": float(big.deg.min()) if len(big) else np.nan})
    pd.DataFrame(chk).to_csv(out / "rule_check.csv", index=False); print(pd.DataFrame(chk).round(3).to_string(index=False))

    # ---------------- synthetic workplaces vs workers ----------------
    wpf = folder / f"{st}_workplace.gpkg"
    if wpf.exists():
        wpl = pyogrio.read_dataframe(wpf, columns=["wp"], read_geometry=False)
        listed = len(wpl); del wpl; gc.collect()
        workers = int((pop.kind == "work").sum())
        rep.update({"workplaces_listed": listed, "workers": workers, "listed_per_worker": listed / workers,
                    "workplaces_used": int(sizes["work"].shape[0]), "share_listed_used": sizes["work"].shape[0] / listed})
    json.dump(rep, open(out / "release_summary.json", "w"), indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float)); print("saved to", out.resolve())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ut")
