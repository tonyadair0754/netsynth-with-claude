"""
calibration_demo.py -- a small teaching experiment: can we recover a generator's parameters from network features?

Setup: a "reference" network is made by a generator with KNOWN parameters (Newman-Watts-Strogatz with k=6, p=0.2).
We pretend we only see its features, then search a grid of (k, p) for the settings whose features are closest.

Lesson (identifiability): each parameter needs a feature that responds to it.
  * clustering alone cannot single out (k, p): several settings give similar clustering;
  * clustering + average path length together pick the true setting.
This is a toy: one generator, 4 replicates per setting, no real data. It illustrates the method (moment matching),
it does not prove it works for realistic networks.

Usage (from the repo root): python experiments/calibration_demo.py       -> results/calibration/calibration_grid.csv
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # make the repo root importable
import paths  # noqa: E402

OUT = paths.RESULTS / "calibration"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n, reps, truth = 500, 4, (6, 0.2)

    def feats(k, p, seed):
        G = nx.newman_watts_strogatz_graph(n, k, p, seed=seed)
        return nx.average_clustering(G), nx.average_shortest_path_length(G)

    ref = np.mean([feats(*truth, 1000 + r) for r in range(reps)], axis=0)
    grid = [(k, p) for k in (2, 4, 6, 8, 10) for p in (0.05, 0.1, 0.2, 0.3, 0.5)]
    table = {kp: np.mean([feats(*kp, 2000 + 10 * i + r) for r in range(reps)], axis=0) for i, kp in enumerate(grid)}
    sd = np.array(list(table.values())).std(axis=0)
    rows = [{"k": kp[0], "p": kp[1], "clustering": v[0], "avg_path": v[1],
             "dist_clustering_only": abs(v[0] - ref[0]) / sd[0],
             "dist_both": float(np.sqrt(((v - ref) / sd) ** 2).sum())} for kp, v in table.items()]
    d = pd.DataFrame(rows); d.to_csv(OUT / "calibration_grid.csv", index=False)
    print(f"truth (k,p) = {truth}; reference clustering {ref[0]:.3f}, average path {ref[1]:.3f}")
    cols = ["k", "p", "clustering", "avg_path"]
    print("\nclosest 4 using clustering ONLY:\n", d.nsmallest(4, "dist_clustering_only")[cols + ["dist_clustering_only"]].round(3).to_string(index=False))
    print("\nclosest 4 using clustering + path length:\n", d.nsmallest(4, "dist_both")[cols + ["dist_both"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
