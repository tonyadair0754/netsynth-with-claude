"""
run_feature_pipeline.py -- two controlled checks of the feature -> synthetic-network pipeline.

CHECK 1 (structure recovery, ground truth known)
    Reference networks from KNOWN generators -> extract features -> synthesize using ONLY
    degree sequence (exact) + average clustering (greedy) -> compare features that were NOT
    targeted (modularity, path length, diameter, spectrum, assortativity, ...).
    Question: which features does "degree + clustering" implicitly recover, and which not?

CHECK 2 (SES mixing as an attribute-assignment problem)
    On each fixed topology, fit node SES labels to the Social Capital Atlas target.
    Question: how well can mixing be matched with ZERO change to topology, and how much
    freedom does the mixing matrix leave (do equally good fits differ in unconstrained features)?

Everything is seeded; parameters are written to run_config.json.
Usage (from the repo root): python experiments/run_feature_pipeline.py [path/to/atlas.csv]   (default: data/atlas/100_x_100_friendship_matrix.csv)
Outputs: results/pipeline/
"""
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # make the repo root importable
import netsynth as ns          # noqa: E402
import paths                   # noqa: E402
from atlas import load_decile_matrix, resolve_atlas, undirected_target   # noqa: E402

OUT = paths.RESULTS / "pipeline"
N, K, REPS, SEED = 500, 10, 3, 2026
MATCH_STEPS, MATCH_TOL = 400_000, 1e-3
LABEL_STEPS = 100_000
NONID_SEEDS = 8
HELD_OUT = ["transitivity", "louvain_modularity", "avg_path_lcc", "diameter_lcc", "lambda2", "rho",
            "degree_assortativity", "lcc_fraction"]


def triangle_homogeneity(G, labels):
    """Fraction of triangles whose three nodes share one SES group (NOT constrained by the target)."""
    tot = same = 0
    adj = {v: set(G[v]) for v in G}
    for u, v in G.edges():
        if u > v:
            u, v = v, u
        for w in adj[u] & adj[v]:
            if w > v:
                tot += 1; same += labels[u] == labels[v] == labels[w]
    return same / tot if tot else np.nan


# ---------------- reference networks with known generators ----------------
def make_reference(kind, seed):
    if kind == "WS(k=8,p=0.1)":
        return nx.watts_strogatz_graph(N, 8, 0.1, seed=seed)
    if kind == "BA(m=4)":
        return nx.barabasi_albert_graph(N, 4, seed=seed)
    if kind == "SBM(4x125,0.15/0.01)":
        p = [[0.15 if i == j else 0.01 for j in range(4)] for i in range(4)]
        return nx.stochastic_block_model([N // 4] * 4, p, seed=seed)
    raise ValueError(kind)


REFS = ["WS(k=8,p=0.1)", "BA(m=4)", "SBM(4x125,0.15/0.01)"]


def random_labels(seed):
    lab = list(np.repeat(np.arange(K), N // K))
    random.Random(seed).shuffle(lab)
    return lab


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = undirected_target(load_decile_matrix(resolve_atlas(sys.argv[1] if len(sys.argv) > 1 else None)))
    pd.DataFrame(target).to_csv(OUT / "ses_target.csv", index=False)
    json.dump({"N": N, "K": K, "REPS": REPS, "SEED": SEED, "MATCH_STEPS": MATCH_STEPS, "MATCH_TOL": MATCH_TOL,
               "LABEL_STEPS": LABEL_STEPS, "NONID_SEEDS": NONID_SEEDS, "refs": REFS,
               "networkx": nx.__version__, "numpy": np.__version__}, open(OUT / "run_config.json", "w"), indent=1)

    rec_rows, fit_rows, nonid_rows, trace = [], [], [], None
    for rep in range(1, REPS + 1):
        s = SEED + rep
        for ri, kind in enumerate(REFS):
            t0 = time.time()
            ref = ns.to_simple_undirected(make_reference(kind, s + 10 * ri))
            ref = nx.convert_node_labels_to_integers(ref)
            fr = ns.extract_features(ref, seed=s)
            # ---- CHECK 1: synthesize from degree sequence + clustering only
            st = ns.initial_simple_graph(fr["degree_sequence"], seed=s + 200)
            c_init = st.avg_clustering()
            c_got, used = ns.match_clustering(st, fr["avg_clustering"], MATCH_STEPS, random.Random(s + 300), MATCH_TOL)
            syn = st.to_graph()
            fs = ns.extract_features(syn, seed=s)
            row = {"rep": rep, "reference": kind, "clustering_target": fr["avg_clustering"],
                   "clustering_start": c_init, "clustering_got": c_got, "proposals_used": used,
                   "degree_ks": ns.degree_ks(fr["degree_sequence"], fs["degree_sequence"])}
            for f in HELD_OUT:
                row[f"ref_{f}"] = fr.get(f, np.nan); row[f"syn_{f}"] = fs.get(f, np.nan)
            rec_rows.append(row)
            # ---- CHECK 2: fit SES labels on the reference topology and on its synthetic twin
            for gname, G in (("reference", ref), ("synthetic", syn)):
                state = ns.SwapState(G)
                lab0 = random_labels(s + 100)
                r_rand = ns._cost(ns.counts_from_edges(state.edges, lab0, K), target)
                L, best, hist = ns.fit_labels(state, target, lab0, LABEL_STEPS, random.Random(s + 400 + ri))
                lab = dict(enumerate(L))
                fe = ns.extract_features(G, lab, K, seed=s)
                fit_rows.append({"rep": rep, "reference": kind, "graph": gname,
                                 "clustering": fe["avg_clustering"], "rmse_random_labels": r_rand,
                                 "rmse_fitted": best, "attr_assortativity": fe["attr_assortativity"],
                                 "attr_modularity": fe["attr_modularity"],
                                 "triangle_homogeneity": triangle_homogeneity(G, lab),
                                 "louvain_modularity": fe["louvain_modularity"], "rho": fe["rho"]})
                if trace is None:
                    trace = hist
            print(f"rep {rep} {kind}: C target {fr['avg_clustering']:.3f} -> got {c_got:.3f} "
                  f"({used} proposals); {time.time() - t0:.0f}s", flush=True)

    # ---- non-identifiability: many equally-good label fits on ONE topology (WS reference, rep 1)
    ref = nx.convert_node_labels_to_integers(make_reference(REFS[0], SEED + 1))
    state = ns.SwapState(ref)
    for k in range(NONID_SEEDS):
        lab0 = random_labels(9000 + k)
        L, best, _ = ns.fit_labels(state, target, lab0, LABEL_STEPS, random.Random(9100 + k))
        lab = dict(enumerate(L))
        parts = [{v for v in ref if lab[v] == g} for g in range(K)]
        nonid_rows.append({"solution": k, "rmse": best, "attr_modularity": nx.community.modularity(ref, parts),
                           "triangle_homogeneity": triangle_homogeneity(ref, lab),
                           "mean_group_ring_spread": float(np.mean([np.std([v for v in p]) for p in parts]))})
        print(f"non-id solution {k}: rmse {best:.4f}", flush=True)

    rec, fit, nonid = pd.DataFrame(rec_rows), pd.DataFrame(fit_rows), pd.DataFrame(nonid_rows)
    rec.to_csv(OUT / "check1_recovery_by_rep.csv", index=False)
    fit.to_csv(OUT / "check2_ses_fit_by_rep.csv", index=False)
    nonid.to_csv(OUT / "check2_nonidentifiability.csv", index=False)

    # ---- summaries (explicit columns; replicate ids are never averaged as features)
    lines = []
    for kind, g in rec.groupby("reference"):
        for f in HELD_OUT:
            a, b = g[f"ref_{f}"], g[f"syn_{f}"]
            lines.append({"reference": kind, "feature": f, "ref_mean": a.mean(), "syn_mean": b.mean(),
                          "syn_sd": b.std(), "mean_abs_diff": (a - b).abs().mean(),
                          "rel_err_of_means": (b.mean() - a.mean()) / abs(a.mean()) if a.mean() else np.nan})
        lines.append({"reference": kind, "feature": "clustering(TARGETED)", "ref_mean": g.clustering_target.mean(),
                      "syn_mean": g.clustering_got.mean(), "syn_sd": g.clustering_got.std(),
                      "mean_abs_diff": (g.clustering_target - g.clustering_got).abs().mean(),
                      "rel_err_of_means": (g.clustering_got.mean() - g.clustering_target.mean()) / g.clustering_target.mean()})
        lines.append({"reference": kind, "feature": "degree_ks(TARGETED)", "ref_mean": 0, "syn_mean": g.degree_ks.mean(),
                      "syn_sd": g.degree_ks.std(), "mean_abs_diff": g.degree_ks.mean(), "rel_err_of_means": np.nan})
    s1 = pd.DataFrame(lines); s1.to_csv(OUT / "check1_recovery_summary.csv", index=False)
    cols = ["clustering", "rmse_random_labels", "rmse_fitted", "attr_assortativity", "attr_modularity",
            "triangle_homogeneity", "louvain_modularity", "rho"]
    s2 = fit.groupby(["reference", "graph"])[cols].agg(["mean", "std"])
    s2.to_csv(OUT / "check2_ses_fit_summary.csv")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\nCHECK 1 -- feature recovery from degree sequence + clustering (n=%d reps)" % REPS)
    print(s1.round(4).to_string(index=False))
    print("\nCHECK 2 -- SES label fit on fixed topology (mean over reps)")
    print(fit.groupby(["reference", "graph"])[cols].mean().round(4).to_string())
    print("\nCHECK 2b -- equally good fits, unconstrained features (%d independent solutions)" % NONID_SEEDS)
    print(nonid.describe().loc[["mean", "std", "min", "max"]].round(4).to_string())

    h = pd.DataFrame(trace)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    ax[0].semilogy(h.step, h.best, label="best"); ax[0].semilogy(h.step, h.current, label="current")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("mixing RMSE"); ax[0].legend()
    ax[1].plot(h.step, h.accept_rate); ax[1].set_xlabel("step"); ax[1].set_ylabel("acceptance rate (window)")
    fig.suptitle("Label-fit annealing trace (first graph)"); fig.savefig(OUT / "label_fit_trace.png", dpi=150)
    print(f"\nSaved to {OUT.resolve()}")


if __name__ == "__main__":
    main()
