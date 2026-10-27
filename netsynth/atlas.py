"""
atlas.py -- turn the Social Capital Atlas friending matrix into the SES mixing target we fit to.

Input: a CSV with column `own_ses_percentile` and 100 columns `friend_prob_p1` ... `friend_prob_p100`
(row = a person's own SES percentile, columns = probability that a friend sits in each SES percentile).
Get it from https://data.humdata.org/dataset/social-capital-atlas (file `friending_rates.csv` there; any filename works here).

Two steps:
  load_decile_matrix : 100x100 -> 10x10 decile matrix (average rows within a decile, sum columns within a decile)
  undirected_target  : make it symmetric and doubly stochastic (a MODELLING choice: an undirected graph needs the two
                       directions of every group-to-group tie to agree; this is an approximation, not the raw data)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import paths


def resolve_atlas(arg=None) -> Path:
    """Use the path given on the command line, else data/atlas/100_x_100_friendship_matrix.csv; exit with help if missing."""
    p = Path(arg) if arg else paths.DEFAULT_ATLAS
    if not p.exists():
        sys.exit(f"Atlas CSV not found: {p}\n"
                 f"Download friending_rates.csv from https://data.humdata.org/dataset/social-capital-atlas and either pass its path "
                 f"as the first argument or save it as {paths.DEFAULT_ATLAS}")
    return p


def load_decile_matrix(path):
    df = pd.read_csv(path).sort_values("own_ses_percentile")
    raw = df[[f"friend_prob_p{i}" for i in range(1, 101)]].to_numpy(float)
    assert raw.shape == (100, 100) and np.isfinite(raw).all() and (raw >= 0).all()
    return np.array([[raw[i:i + 10, j:j + 10].sum(axis=1).mean() for j in range(0, 100, 10)]
                     for i in range(0, 100, 10)])


def undirected_target(P):
    """Symmetrize + Sinkhorn to a symmetric doubly-stochastic matrix (a MODELLING choice, see notes)."""
    A = np.maximum((P + P.T) / 2, 1e-12)
    u = v = np.ones(len(A))
    for _ in range(20000):
        u = 1.0 / (A @ v); v = 1.0 / (A.T @ u)
    Q = u[:, None] * A * v[None, :]
    Q = (Q + Q.T) / 2
    for _ in range(1000):
        s = 1.0 / np.sqrt(np.maximum(Q.sum(axis=1), 1e-15))
        Q = s[:, None] * Q * s[None, :]
    return Q


