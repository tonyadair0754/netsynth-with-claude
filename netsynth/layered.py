"""
layered.py -- spec-driven, seeded generator for LAYERED group-membership networks
(household / work / school / daycare / ...), plus the metric helpers shared with analyze_release.py.

What we learned from the Utah release (see analyze_release.py) and built in here:
  1. A layer is "who is in which group" + "how people inside a group are wired".
     Wiring rule of the original repo: clique if the group has <= 5 members, otherwise a
     Newman-Watts-Strogatz ring (k=4, p=0.3). Verified exactly on the Utah release.
  2. Because of that rule, degree barely depends on group size (mean ~5.2 in any large group),
     so group SIZES mainly control connectivity (components, path length), not degree.
  3. Group-size distributions are INPUTS: take them from data (empirical pmf), or sweep them
     as a sensitivity analysis. The Utah release's own work/daycare sizes look implausible
     (83% one-person workplaces; daycares of thousands), so we do not treat them as ground truth.
  4. Different empirical statistics describe different layers. The Social Capital Atlas describes
     FRIENDSHIPS, so SES mixing is matched on a friendship layer, not on household/work/school edges.
Everything takes an explicit seed.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

AGE_BINS, AGE_NAMES = [5, 18, 35, 50, 65], ["0-4", "5-17", "18-34", "35-49", "50-64", "65+"]


# ------------------------------------------------------------------ shared metric helpers
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def clustering_sample(A, S):
    """Local clustering for sampled nodes S (NetworkX convention: degree < 2 -> 0)."""
    As = A[S]
    tri = (As @ A).multiply(As).sum(axis=1).A1 / 2
    k = np.asarray(As.sum(axis=1)).ravel()
    return np.where(k > 1, 2 * tri / np.maximum(k * (k - 1), 1), 0.0), k


def age_mixing(A, age_group, K=6):
    A = A.tocoo(); m = A.row < A.col
    r, c = A.row[m], A.col[m]
    C = np.zeros((K, K)); np.add.at(C, (age_group[r], age_group[c]), 1); C = C + C.T
    P = C / np.maximum(C.sum(1, keepdims=True), 1)
    e = C / C.sum(); a = e.sum(1); denom = 1 - (a * a).sum()
    return P, float((np.trace(e) - (a * a).sum()) / denom) if denom > 1e-12 else np.nan


def to_csr(edges, n):
    """(E,2) integer array -> symmetric 0/1 CSR matrix."""
    if len(edges) == 0:
        return sp.csr_matrix((n, n), dtype=np.int8)
    e = np.asarray(edges)
    A = sp.csr_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
    A = A.maximum(A.T); A.data[:] = 1
    return A


# ------------------------------------------------------------------ within-group wiring
def within_group_edges(members, seed, clique_max=5, k=4, p=0.3):
    """Original repository rule: clique if n <= clique_max, else Newman-Watts-Strogatz(n, k, p)."""
    n = len(members)
    if n <= 1:
        return []
    m = np.asarray(members)
    if n <= clique_max:
        a, b = np.triu_indices(n, 1)
        return list(zip(m[a].tolist(), m[b].tolist()))
    g = nx.newman_watts_strogatz_graph(n, k, p, seed=int(seed))
    return [(int(m[a]), int(m[b])) for a, b in g.edges()]


def edges_for_groups(groups, seed, **kw):
    out, s = [], int(seed)
    for g in groups:
        s += 1
        out.extend(within_group_edges(g, s, **kw))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


# ------------------------------------------------------------------ group membership
def partition_by_size_pmf(members, sizes, probs, rng, cap=None):
    """
    Cut `members` (shuffled) into consecutive groups whose sizes are drawn from an EMPIRICAL pmf
    (`sizes`, `probs`). `cap` truncates any drawn size (a crude capacity constraint). The last group
    takes whatever remains.
    """
    members = np.array(members); rng.shuffle(members)
    sizes = np.asarray(sizes); probs = np.asarray(probs, float); probs = probs / probs.sum()
    groups, i, n = [], 0, len(members)
    draws = rng.choice(sizes, size=max(n, 1), p=probs)       # more draws than groups can ever need
    j = 0
    while i < n:
        s = int(draws[j]); j += 1
        if cap is not None:
            s = min(s, int(cap))
        groups.append(members[i:i + s].tolist()); i += s
    return groups


def partition_lognormal_weights(members, n_groups, rng, mean=2.0, sigma=1.0):
    """Repository mechanism (tools.wp_proba): each group gets a lognormal weight; each person picks a group ~ weight."""
    members = np.asarray(members)
    w = rng.lognormal(mean, sigma, size=n_groups); w /= w.sum()
    choice = rng.choice(n_groups, size=len(members), p=w)
    order = np.argsort(choice, kind="stable")
    sc = choice[order]; cuts = np.flatnonzero(np.diff(sc)) + 1
    return [members[order][a:b].tolist() for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(sc)])]


# ------------------------------------------------------------------ population from a household template
def population_from_household_template(template, n_target, rng):
    """
    template: DataFrame with columns hh, age (one row per person; whole households).
    Resample WHOLE households (with replacement) until >= n_target people, so household size and
    household age composition are inherited from the template. Returns DataFrame(age, hh).
    """
    g = template.groupby("hh").age.apply(np.asarray)
    ids = g.index.to_numpy(); sizes = g.map(len).to_numpy()
    ages, hh, tot, h = [], [], 0, 0
    order = rng.integers(0, len(ids), size=int(n_target / sizes.mean() * 1.3) + 50)
    for o in order:
        ages.append(g.iloc[o]); hh.extend([h] * sizes[o]); tot += sizes[o]; h += 1
        if tot >= n_target:
            break
    return pd.DataFrame({"age": np.concatenate(ages), "hh": hh})


def build_layers(pop, cfg, seed):
    """
    pop: DataFrame(age, hh). cfg keys:
      employment_rate, school_pmf=(sizes,probs), daycare_pmf=(sizes,probs), work_pmf=(sizes,probs) OR work_mean_size,
      school_cap, daycare_cap (optional).
    Returns dict layer -> CSR adjacency and dict layer -> list of groups.
    """
    rng = np.random.default_rng(seed); n = len(pop); age = pop.age.to_numpy()
    kids = np.flatnonzero((age >= 5) & (age < 18)); tiny = np.flatnonzero(age < 5)
    adults = np.flatnonzero(age >= 18)
    workers = rng.choice(adults, size=int(round(cfg["employment_rate"] * len(adults))), replace=False)
    hh_groups = [g.tolist() for _, g in pd.Series(np.arange(n)).groupby(pop.hh.to_numpy())]
    groups = {"household": hh_groups,
              "school": partition_by_size_pmf(kids, *cfg["school_pmf"], rng, cfg.get("school_cap")),
              "daycare": partition_by_size_pmf(tiny, *cfg["daycare_pmf"], rng, cfg.get("daycare_cap"))}
    if "work_mean_size" in cfg:
        groups["work"] = partition_lognormal_weights(workers, max(int(len(workers) / cfg["work_mean_size"]), 1), rng)
    else:
        groups["work"] = partition_by_size_pmf(workers, *cfg["work_pmf"], rng)
    layers = {k: to_csr(edges_for_groups(g, seed * 1000 + 7 * i), n) for i, (k, g) in enumerate(groups.items())}
    return layers, groups


def merge_layers(layers):
    M = sum(m.astype(np.int8) for m in layers.values())
    return (M > 0).astype(np.int8).tocsr()


# ------------------------------------------------------------------ friendship layer with a prescribed mixing matrix
def mixing_configuration_model(degrees, labels, Q, rng):
    """
    Configuration-model-style stub matching with a PRESCRIBED group mixing matrix.
      degrees : desired degree per node;  labels : group per node;  Q : row-stochastic target P(friend group | own group).
    Method: group i has D_i stubs; the symmetrised target number of i--j stub pairs is C_ij = (D_i Q_ij + D_j Q_ji)/2
    (scaled down if a group would need more stubs than it has); stubs of i earmarked for j are paired
    with stubs of j earmarked for i. Self-loops and duplicate pairs are dropped, so realised degrees are
    slightly below `degrees`. Returns an (E,2) array.
    """
    labels = np.asarray(labels); K = Q.shape[0]; degrees = np.asarray(degrees, int)
    D = np.array([degrees[labels == i].sum() for i in range(K)], float)
    C = (D[:, None] * Q + (D[None, :] * Q.T)) / 2
    for _ in range(50):                                   # keep row sums <= D_i, symmetric scaling
        s = np.minimum(1.0, D / np.maximum(C.sum(1), 1e-12)); C = C * np.sqrt(np.outer(s, s))
    m = np.rint(C).astype(int); m = np.triu(m) + np.triu(m, 1).T
    for i in range(K):
        m[i, i] -= m[i, i] % 2                            # within-group stubs come in pairs
    stubs = []
    for i in range(K):
        nodes = np.flatnonzero(labels == i)
        st = np.repeat(nodes, degrees[nodes]); rng.shuffle(st); stubs.append(st)
    ptr = [0] * K; pairs = []
    for i in range(K):
        for j in range(i, K):
            c = int(m[i, j])
            if c <= 0:
                continue
            if i == j:
                c = min(c, (len(stubs[i]) - ptr[i]) // 2 * 2)
                a = stubs[i][ptr[i]:ptr[i] + c]; ptr[i] += c
                pairs.append(np.column_stack([a[0::2], a[1::2]]))
            else:
                c = min(c, len(stubs[i]) - ptr[i], len(stubs[j]) - ptr[j])
                a = stubs[i][ptr[i]:ptr[i] + c]; b = stubs[j][ptr[j]:ptr[j] + c]; ptr[i] += c; ptr[j] += c
                pairs.append(np.column_stack([a, b]))
    e = np.vstack(pairs) if pairs else np.zeros((0, 2), int)
    e = e[e[:, 0] != e[:, 1]]; e = np.sort(e, axis=1)
    return np.unique(e, axis=0)
