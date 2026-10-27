"""
run_utah_replication.py -- three controlled experiments built on what the Utah release taught us.

EXP 1  REPLICATION  Does OUR code implement the same mechanism as the repository? Fed ONLY group-size histograms +
                    a household template from the Utah release, can layered.py reproduce the release's features?
                    This is an IMPLEMENTATION check and is circular by design: the release is itself the output of
                    that mechanism, so agreement says NOTHING about realism. Differences point to ingredients we left
                    out (mainly geography). Realism needs observed data: see compare_networks.py.
EXP 2  SENSITIVITY  Group sizes are inputs. How much do network features move when the workplace / daycare size
                    assumptions change? (One knob at a time.)
EXP 3  SES          Combine with the Social Capital Atlas target. Which layer does the Atlas describe?
                    A: individual labels on the whole merged graph (what we did before)
                    B: household members forced to share an SES label, still fitted on the merged graph
                    C: households share SES; SES mixing is imposed on a separate FRIENDSHIP layer
                       (friendship degree = placeholder assumption, NOT estimated from data)

Needs: data/utah_reference/ (committed) and the Atlas CSV. Usage (from the repo root):
    python experiments/run_utah_replication.py [path/to/atlas.csv]      -> results/utah_replication/
All randomness is seeded; parameters are written to run_config_utah.json.
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components, dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # make the repo root importable
import layered as L   # noqa: E402
import netsynth as ns   # noqa: E402
import paths   # noqa: E402
from atlas import load_decile_matrix, resolve_atlas, undirected_target   # noqa: E402

REF, OUT = paths.UTAH_REFERENCE, paths.RESULTS / "utah_replication"
N_TWIN, SEEDS = 300_000, (1, 2, 3)
N_SENS, SENS_SEEDS = 200_000, (1, 2)
N_SES, SES_SEEDS, FIT_STEPS = 60_000, (1, 2, 3), 80_000
FRIEND_MEAN_DEGREE = 6          # PLACEHOLDER assumption for the friendship layer (not estimated from data)
K = 10


def hist_pmf(name):
    d = pd.read_csv(REF / f"size_hist_{name}.csv")
    return d["size"].to_numpy(), d["count"].to_numpy() / d["count"].sum()


def base_cfg():
    emp = json.load(open(REF / "release_summary.json"))["employed_share_of_adults"]
    return {"employment_rate": emp, "school_pmf": hist_pmf("school"), "daycare_pmf": hist_pmf("daycare"),
            "work_pmf": hist_pmf("work")}


def features(layers, age, seed):
    n = len(age); rng = np.random.default_rng(seed); ag = np.digitize(age, L.AGE_BINS)
    M = L.merge_layers(layers)
    d = np.asarray(M.sum(1)).ravel()
    S = rng.choice(n, min(60_000, n), replace=False)
    c, kk = L.clustering_sample(M, S)
    nc, lab = connected_components(M, directed=False); cs = np.bincount(lab)
    gi = np.flatnonzero(lab == cs.argmax()); src = rng.choice(gi, 8, replace=False)
    D = dijkstra(M, directed=False, unweighted=True, indices=src)[:, gi]
    f = {"mean_degree": d.mean(), "max_degree": d.max(), "degree_gini": L.gini(d),
         "clustering_all_nodes": c.mean(), "clustering_deg_ge2": c[kk > 1].mean(),
         "giant_share": cs.max() / n, "isolate_share": (cs == 1).sum() / n,
         "avg_path_giant": D[D > 0].mean(), "age_assortativity": L.age_mixing(M, ag)[1]}
    for k, A in layers.items():
        f[f"mean_degree_{k}"] = A.sum() / n
    return f, M


def exp1():
    cfg, rows = base_cfg(), []
    tmpl = pd.read_csv(REF / "household_sample.csv.gz")
    for s in SEEDS:
        t = time.time(); rng = np.random.default_rng(s)
        pop = L.population_from_household_template(tmpl, N_TWIN, rng)
        layers, _ = L.build_layers(pop, cfg, s)
        f, _ = features(layers, pop.age.to_numpy(), s); f["seed"] = s; rows.append(f)
        print(f"  exp1 seed {s}: N={len(pop):,} {time.time() - t:.0f}s", flush=True)
    twin = pd.DataFrame(rows)
    ref = pd.read_csv(REF / "layer_features.csv", index_col=0)
    refd = {"mean_degree": ref.loc["merged", "mean_degree_all"], "max_degree": ref.loc["merged", "max_degree"],
            "degree_gini": ref.loc["merged", "degree_gini_all"], "clustering_all_nodes": ref.loc["merged", "avg_clustering_all_nodes"],
            "clustering_deg_ge2": ref.loc["merged", "avg_clustering_nodes_deg_ge2"], "giant_share": ref.loc["merged", "giant_share"],
            "isolate_share": ref.loc["merged", "isolates"] / 3_258_959, "avg_path_giant": ref.loc["merged", "avg_path_giant_sampled8"],
            "age_assortativity": ref.loc["merged", "age_assortativity"]}
    for k in ("household", "school", "daycare", "work"):
        refd[f"mean_degree_{k}"] = ref.loc[k, "mean_degree_all"]
    tab = pd.DataFrame({"utah_release": pd.Series(refd), "twin_mean": twin.drop(columns="seed").mean(),
                        "twin_sd": twin.drop(columns="seed").std()}).dropna(subset=["utah_release"])
    tab["rel_diff"] = (tab.twin_mean - tab.utah_release) / tab.utah_release
    twin.to_csv(OUT / "exp1_twin_by_seed.csv", index=False); tab.to_csv(OUT / "exp1_replication.csv")
    print(tab.round(4).to_string())
    return tab


def exp1b_scale():
    """Does the twin's path length simply grow with population size? (tests whether scale explains the Utah gap)"""
    cfg, rows = base_cfg(), []
    tmpl = pd.read_csv(REF / "household_sample.csv.gz")
    for n in (25_000, 50_000, 100_000, 200_000, 400_000, 800_000, 1_600_000):
        rng = np.random.default_rng(7)
        pop = L.population_from_household_template(tmpl, n, rng)
        layers, _ = L.build_layers(pop, cfg, 7)
        f, _ = features(layers, pop.age.to_numpy(), 7); rows.append({"N": len(pop), "avg_path_giant": f["avg_path_giant"],
                                                                      "giant_share": f["giant_share"], "mean_degree": f["mean_degree"]})
        print(f"  scale N={len(pop):,}: path {f['avg_path_giant']:.2f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "exp1b_path_vs_N.csv", index=False); print(df.round(3).to_string(index=False))
    return df


def exp2():
    base = base_cfg(); tmpl = pd.read_csv(REF / "household_sample.csv.gz")
    scen = {"A. Utah pmf (as released)": {},
            "B. work: lognormal groups, mean size 1.3": {"work_mean_size": 1.3},
            "C. work: mean size 5": {"work_mean_size": 5}, "D. work: mean size 15": {"work_mean_size": 15},
            "E. work: mean size 50": {"work_mean_size": 50},
            "F. daycare cap 100": {"daycare_cap": 100}, "G. daycare cap 30": {"daycare_cap": 30},
            "H. school cap 300": {"school_cap": 300}}
    rows = []
    for name, over in scen.items():
        for s in SENS_SEEDS:
            t = time.time(); cfg = {**base, **over}; rng = np.random.default_rng(100 + s)
            pop = L.population_from_household_template(tmpl, N_SENS, rng)
            layers, _ = L.build_layers(pop, cfg, 100 + s)
            f, _ = features(layers, pop.age.to_numpy(), s); f.update({"scenario": name, "seed": s}); rows.append(f)
            print(f"  exp2 {name} seed {s}: {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "exp2_sensitivity_by_seed.csv", index=False)
    cols = ["mean_degree", "max_degree", "clustering_all_nodes", "giant_share", "isolate_share", "avg_path_giant", "mean_degree_work", "mean_degree_daycare"]
    tab = df.groupby("scenario")[cols].mean(); tab.to_csv(OUT / "exp2_sensitivity.csv")
    print(tab.round(3).to_string()); return tab


def ses_rmse(edges_or_csr, labels, target):
    A = edges_or_csr.tocoo(); m = A.row < A.col
    C = np.zeros((K, K)); np.add.at(C, (labels[A.row[m]], labels[A.col[m]]), 1); C = C + C.T
    return ns.mixing_rmse(ns.row_normalize(C), target), C


def exp3(csv):
    target = undirected_target(load_decile_matrix(csv))
    base = base_cfg(); tmpl = pd.read_csv(REF / "household_sample.csv.gz"); rows = []
    for s in SES_SEEDS:
        t = time.time(); rng = np.random.default_rng(200 + s)
        pop = L.population_from_household_template(tmpl, N_SES, rng); n = len(pop)
        layers, groups = L.build_layers(pop, base, 200 + s); M = L.merge_layers(layers)
        hh_units = groups["household"]
        within_hh_edge_share = layers["household"].nnz / M.nnz            # share of merged edge-ends that are household ties
        # balanced starting labels per household unit
        ulab0 = list(np.tile(np.arange(K), len(hh_units) // K + 1)[:len(hh_units)]); random.Random(s).shuffle(ulab0)
        node0 = np.zeros(n, int)
        for u, mem in enumerate(hh_units):
            node0[mem] = ulab0[u]
        r_start, _ = ses_rmse(M, node0, target)
        state = ns.SwapState(nx_from_csr(M))
        # A: individual labels on the merged graph
        lab0 = list(node0); random.Random(s + 1).shuffle(lab0)
        LA, costA, _ = ns.fit_labels(state, target, lab0, FIT_STEPS, random.Random(s + 2))
        # B: household members share a label, fitted on the merged graph
        LB, costB, _ = ns.fit_labels_grouped(state, target, hh_units, ulab0, FIT_STEPS, random.Random(s + 3))
        # C: households share SES (random), mixing imposed on a separate friendship layer
        deg_f = np.random.default_rng(s).poisson(FRIEND_MEAN_DEGREE, n)
        E = L.mixing_configuration_model(deg_f, node0, target, np.random.default_rng(s + 4))
        F = L.to_csr(E, n)
        costC, _ = ses_rmse(F, node0, target)
        merged_with_friends = L.merge_layers({**layers, "friendship": F})
        costC_merged, _ = ses_rmse(merged_with_friends, node0, target)
        ag = pop.age.to_numpy()
        f0, _ = features(layers, ag, s); f1, _ = features({**layers, "friendship": F}, ag, s)
        diag_target = float(np.mean(np.diag(target)))
        rows.append({"seed": s, "N": n, "household_edge_share_of_merged": within_hh_edge_share, "target_mean_diagonal": diag_target,
                     "rmse_random_household_labels": r_start, "A_individual_labels_rmse": costA, "B_household_shared_rmse": costB,
                     "C_friendship_layer_rmse": costC, "C_merged_graph_rmse": costC_merged,
                     "path_without_friends": f0["avg_path_giant"], "path_with_friends": f1["avg_path_giant"],
                     "clustering_without_friends": f0["clustering_all_nodes"], "clustering_with_friends": f1["clustering_all_nodes"],
                     "giant_without": f0["giant_share"], "giant_with": f1["giant_share"],
                     "mean_deg_without": f0["mean_degree"], "mean_deg_with": f1["mean_degree"]})
        print(f"  exp3 seed {s}: {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "exp3_ses_by_seed.csv", index=False)
    tab = df.drop(columns="seed").agg(["mean", "std"]).T; tab.to_csv(OUT / "exp3_ses.csv"); print(tab.round(4).to_string())
    return tab


def nx_from_csr(M):
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(range(M.shape[0]))
    A = M.tocoo(); m = A.row < A.col
    G.add_edges_from(zip(A.row[m].tolist(), A.col[m].tolist()))
    return G


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    csv = resolve_atlas(sys.argv[1] if len(sys.argv) > 1 else None)
    json.dump({"N_TWIN": N_TWIN, "SEEDS": SEEDS, "N_SENS": N_SENS, "SENS_SEEDS": SENS_SEEDS, "N_SES": N_SES,
               "SES_SEEDS": SES_SEEDS, "FIT_STEPS": FIT_STEPS, "FRIEND_MEAN_DEGREE": FRIEND_MEAN_DEGREE},
              open(OUT / "run_config_utah.json", "w"), indent=1)
    print("EXP 1 -- implementation replication of the Utah release (circular by design; NOT a realism test)"); exp1()
    print("\nEXP 1b -- twin path length versus population size"); exp1b_scale()
    print("\nEXP 2 -- sensitivity to group-size assumptions"); exp2()
    print("\nEXP 3 -- SES mixing: which layer?"); exp3(csv)
