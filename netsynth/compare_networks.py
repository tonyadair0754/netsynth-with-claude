"""
compare_networks.py -- the INDEPENDENT validation step: compare a synthetic network against an observed one.

Why this exists: a generator can match demographic totals (or the features we fed it) and still miss the real
social structure. The only way to find out is to compare against an OBSERVED network on features that were NOT
used to build the synthetic one. This script prints/saves a side-by-side feature table plus distribution distances.

Inputs (plain text edge lists, one edge per line: "u v" separated by spaces, tabs or commas; '#' lines ignored):
    python compare_networks.py --observed facebook_combined.txt --synthetic my_synthetic_edges.txt
Optional node labels (CSV "node,label", labels = integers 0..K-1; same K for both) add a mixing-matrix comparison:
    --obs-labels obs_labels.csv --syn-labels syn_labels.csv
Try it with no data:   python compare_networks.py --demo      (two made-up graphs; NEITHER is real data)

How to read the output
  * 'flag' marks features that differ by more than --tol (default 25%). It is a prompt to look, not a verdict.
  * Features marked size_sensitive (path length, diameter, component counts, spectrum, max degree) change with the
    number of nodes, so they are only comparable when both networks have similar size.
  * Matching the observed network on ALL these features still does not prove the generator is right for other
    populations/contexts; it shows it reproduces THIS observed network along these dimensions.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import netsynth as ns
import paths

SIZE_SENSITIVE = {"nodes", "edges", "components", "avg_path_lcc", "diameter_lcc", "diameter_lcc_lowerbound", "lambda2", "rho",
                  "lazy_gap", "degree_max", "louvain_communities", "lambda_min"}
MAX_NODES = 300_000


def read_edgelist(path) -> nx.Graph:
    edges = []
    for i, line in enumerate(open(path)):
        line = line.strip()
        if not line or line[0] in "#%":
            continue
        t = [x for x in re.split(r"[,\s;]+", line) if x != ""]
        if len(t) < 2:
            continue
        if i == 0 and t[0].lower() in {"source", "src", "from", "node1", "u", "id", "a"}:
            continue                                                    # header row
        edges.append((t[0], t[1]))
    G = nx.Graph(); G.add_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_nodes() > MAX_NODES:
        sys.exit(f"{path}: {G.number_of_nodes():,} nodes > {MAX_NODES:,}; sample a subgraph first (this tool is for exploration-sized graphs).")
    return G


def read_labels(path, G):
    df = pd.read_csv(path)
    lab = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(int)))
    missing = [v for v in G if str(v) not in lab]
    if missing:
        sys.exit(f"{path}: {len(missing)} nodes have no label (e.g. {missing[:3]})")
    return {v: lab[str(v)] for v in G}


def component_sizes(G):
    return np.array(sorted((len(c) for c in nx.connected_components(G)), reverse=True))


def clustering_by_degree(G, bins=(1, 2, 4, 8, 16, 10**9)):
    deg = dict(G.degree()); cl = nx.clustering(G)
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        v = [cl[n] for n in G if lo <= deg[n] < hi]
        out[f"clustering_deg_{lo}-{hi - 1 if hi < 10**9 else 'inf'}"] = float(np.mean(v)) if v else np.nan
    return out


def profile(G, labels=None, K=None, seed=0):
    f = ns.extract_features(G, labels, K, seed=seed)
    cs = component_sizes(G); deg = np.array([d for _, d in G.degree()])
    f["isolate_share"] = float((deg == 0).mean()); f["giant_share"] = f["lcc_fraction"]
    f["median_component_size"] = float(np.median(cs)); f["n_components_ge_2"] = int((cs >= 2).sum())
    f.update(clustering_by_degree(G))
    return f


def compare(obs: nx.Graph, syn: nx.Graph, obs_lab=None, syn_lab=None, tol=0.25, outdir=Path("compare_results"), seed=0):
    outdir.mkdir(parents=True, exist_ok=True)
    K = None
    if obs_lab is not None and syn_lab is not None:
        K = max(max(obs_lab.values()), max(syn_lab.values())) + 1
    fo = profile(obs, obs_lab, K, seed); fs = profile(syn, syn_lab, K, seed)
    rows = []
    for k in fo:
        if k in fs and np.isscalar(fo[k]) and np.isscalar(fs[k]):
            a, b = float(fo[k]), float(fs[k])
            rel = (b - a) / abs(a) if a not in (0, np.nan) and np.isfinite(a) and a != 0 else np.nan
            rows.append({"feature": k, "observed": a, "synthetic": b, "abs_diff": b - a, "rel_diff": rel,
                         "flag": "check" if (np.isfinite(rel) and abs(rel) > tol) else "", "size_sensitive": k in SIZE_SENSITIVE})
    dist = {"degree_KS": ns.degree_ks(fo["degree_sequence"], fs["degree_sequence"]),
            "component_size_KS": ns.degree_ks(component_sizes(obs), component_sizes(syn))}
    if K is not None:
        dist["mixing_matrix_RMSE"] = ns.mixing_rmse(fo["mixing_matrix"], fs["mixing_matrix"])
    tab = pd.DataFrame(rows); tab.to_csv(outdir / "feature_comparison.csv", index=False)
    pd.Series(dist).to_csv(outdir / "distribution_distances.csv", header=["value"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for G, name, c in ((obs, "observed", "C0"), (syn, "synthetic", "C1")):
        d = np.sort([x for _, x in G.degree() if x > 0]); ax[0].loglog(d, 1 - np.arange(len(d)) / len(d), label=name, color=c)
        s = np.sort(component_sizes(G)); ax[1].loglog(s, 1 - np.arange(len(s)) / len(s), label=name, color=c)
    ax[0].set_xlabel("degree"); ax[0].set_ylabel("P(D >= d)"); ax[0].legend(); ax[0].set_title("Degree distribution (CCDF)")
    ax[1].set_xlabel("component size"); ax[1].set_ylabel("P(S >= s)"); ax[1].set_title("Component sizes (CCDF)")
    fig.savefig(outdir / "distributions.png", dpi=150)
    if K is not None:
        pd.DataFrame(fo["mixing_matrix"]).to_csv(outdir / "mixing_observed.csv", index=False)
        pd.DataFrame(fs["mixing_matrix"]).to_csv(outdir / "mixing_synthetic.csv", index=False)
    return tab, dist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--observed"); ap.add_argument("--synthetic"); ap.add_argument("--obs-labels"); ap.add_argument("--syn-labels")
    ap.add_argument("--tol", type=float, default=0.25); ap.add_argument("--out", default=None, help="output folder (default: ./compare_results, or results/compare_demo for --demo)")
    ap.add_argument("--demo", action="store_true", help="compare two made-up graphs (not real data) to show the output format")
    a = ap.parse_args()
    if a.demo:
        import layered as L
        obs = nx.powerlaw_cluster_graph(3000, 3, 0.4, seed=1)                      # stand-in: hubs + clustering
        rng = np.random.default_rng(1)
        pop = pd.DataFrame({"age": rng.integers(0, 85, 3000), "hh": np.sort(rng.integers(0, 1100, 3000))})
        sizes = np.array([1, 2, 3, 4, 5]); probs = np.array([.2, .3, .2, .2, .1])
        layers, _ = L.build_layers(pop, {"employment_rate": .6, "school_pmf": ([300], [1.0]), "daycare_pmf": ([40], [1.0]),
                                         "work_pmf": (sizes, probs)}, 1)
        syn = nx.from_scipy_sparse_array(L.merge_layers(layers))
        print("DEMO: 'observed' = a power-law-cluster graph, 'synthetic' = a group-membership graph. NEITHER is real data.\n")
    else:
        if not (a.observed and a.synthetic):
            ap.error("give --observed and --synthetic edge lists (or --demo)")
        obs, syn = read_edgelist(a.observed), read_edgelist(a.synthetic)
    ol = sl = None
    if a.obs_labels and a.syn_labels:
        ol, sl = read_labels(a.obs_labels, obs), read_labels(a.syn_labels, syn)
    out = Path(a.out) if a.out else (paths.RESULTS / "compare_demo" if a.demo else Path("compare_results"))
    tab, dist = compare(obs, syn, ol, sl, a.tol, out)
    pd.set_option("display.width", 200)
    print(tab.round(4).to_string(index=False)); print("\nDistribution distances (0 = identical):")
    for k, v in dist.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
