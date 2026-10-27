"""
netsynth.py -- feature extraction from a network + feature-targeted synthetic generation.

Design principle (learned from diagnosing ses_network_experiment v3):
  Generate STRUCTURE and ATTRIBUTES in separate stages so that each feature has a lever
  that does not disturb the others.

    Stage 1  topology   : degree sequence (exact) + clustering (greedy, degree-preserving swaps)
    Stage 2  attributes : assign group labels to nodes on a FIXED topology to match a mixing matrix
                          (topology untouched => degree, clustering, paths, spectrum unchanged)
    Stage 3  (optional) : degree-preserving edge swaps toward the mixing target under a HARD
                          cap on how far clustering may move (constrained, not penalized)

Conventions
  * Graphs handed to the generator are simple, undirected, nodes labelled 0..n-1.
  * Exchange matrix W = row-normalized adjacency (the lab notes' build_weight_matrix);
    dynamics x(t+1) = W^T x(t). NOT lazy. Isolated nodes get a self-loop.
  * Mixing matrix P[i, j] = P(neighbor in group j | node in group i); counts count every
    undirected edge in both directions, so a within-group edge adds 2 to the diagonal.
"""
from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np


# --------------------------------------------------------------------------- utilities
def gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if x.size == 0 or np.allclose(x, 0):
        return 0.0
    if (x < 0).any():
        raise ValueError("Gini requires nonnegative values.")
    n = x.size
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def to_simple_undirected(G) -> nx.Graph:
    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


# --------------------------------------------------------------------------- mixing
def mixing_counts(G, labels, K) -> np.ndarray:
    C = np.zeros((K, K))
    for u, v in G.edges():
        i, j = labels[u], labels[v]
        C[i, j] += 1
        C[j, i] += 1
    return C


def row_normalize(C) -> np.ndarray:
    return C / np.maximum(C.sum(axis=1, keepdims=True), 1)


def mixing_rmse(P, target) -> float:
    return float(np.sqrt(np.mean((P - target) ** 2)))


def attribute_assortativity(C) -> float:
    """Newman's assortativity coefficient from a symmetric count matrix."""
    e = C / C.sum()
    a = e.sum(axis=1)
    s = float(np.sum(a * a))
    return float((np.trace(e) - s) / (1 - s))


# --------------------------------------------------------------------------- exchange model
def weight_matrix(G, nodelist=None) -> np.ndarray:
    """Row-normalized adjacency (lab notes). Works for directed graphs too."""
    A = nx.to_numpy_array(G, nodelist=nodelist, dtype=float)
    rs = A.sum(axis=1)
    for i in range(len(A)):
        if rs[i] > 0:
            A[i] /= rs[i]
        else:
            A[i, i] = 1.0
    return A


def simulate(W, x0, eps=1e-6, max_iter=10000) -> dict:
    """Iterate x <- W^T x until ||x(t+1)-x(t)||_2 < eps (paper Eq. 3)."""
    x = np.asarray(x0, float).copy()
    total = x.sum()
    for t in range(1, max_iter + 1):
        nxt = W.T @ x
        if not np.isclose(nxt.sum(), total, rtol=1e-9, atol=1e-9):
            raise RuntimeError("conservation violated")
        done = np.linalg.norm(nxt - x) < eps
        x = nxt
        if done:
            return {"x": x, "steps": t, "converged": True}
    return {"x": x, "steps": max_iter, "converged": False}


def stationary_by_degree(G, x0, nodelist) -> np.ndarray:
    """Undirected graphs: within each component, x* is proportional to degree."""
    idx = {v: i for i, v in enumerate(nodelist)}
    x0 = np.asarray(x0, float)
    out = np.zeros(len(nodelist))
    for comp in nx.connected_components(G):
        ii = [idx[v] for v in comp]
        d = np.array([G.degree(v) for v in comp], float)
        out[ii] = x0[ii].sum() * d / d.sum() if d.sum() > 0 else x0[ii]
    return out


def lcc_spectrum(G, dense_limit=3000) -> dict:
    """
    Spectrum of W on the largest component (undirected). W is similar to the symmetric matrix
    D^-1/2 A D^-1/2, so eigenvalues are real. rho = max(|lambda_2|, |lambda_min|) is the
    asymptotic per-step contraction of the NON-lazy dynamics (rho -> 1 means slow / no convergence;
    lambda_min = -1 means bipartite, i.e. the iteration oscillates forever).
    Components larger than `dense_limit` nodes use sparse eigensolvers (needed for real-size networks).
    """
    comp = max(nx.connected_components(G), key=len)
    L = G.subgraph(comp)
    nodes = list(L.nodes())
    if len(nodes) <= dense_limit:
        A = nx.to_numpy_array(L, nodelist=nodes)
        d = A.sum(axis=1)
        S = A / np.sqrt(np.outer(d, d))
        ev = np.sort(np.linalg.eigvalsh(S))[::-1]
        lam2, lmin = float(ev[1]), float(ev[-1])
    else:
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        A = nx.to_scipy_sparse_array(L, nodelist=nodes, format="csr", dtype=float)
        d = np.asarray(A.sum(axis=1)).ravel()
        Dm = sp.diags(1.0 / np.sqrt(d))
        S = Dm @ A @ Dm
        top = np.sort(eigsh(S, k=2, which="LA", return_eigenvectors=False, tol=1e-8))[::-1]
        lam2 = float(top[1])
        lmin = float(eigsh(S, k=1, which="SA", return_eigenvectors=False, tol=1e-8)[0])
    return {"lambda2": lam2, "lambda_min": lmin, "rho": max(abs(lam2), abs(lmin)),
            "lazy_gap": (1 - lam2) / 2}


# --------------------------------------------------------------------------- feature extraction
def extract_features(G, labels=None, K=None, seed=0, path_sources=300) -> dict:
    """
    Feature profile of a network (+ optional node group labels: dict/array node -> 0..K-1).
    Scalars are floats; arrays: degree_sequence, mixing_matrix, mixing_counts, group_sizes.
    """
    H = to_simple_undirected(G)
    n, m = H.number_of_nodes(), H.number_of_edges()
    deg = np.array([d for _, d in H.degree()], float)
    comps = list(nx.connected_components(H))
    lcc = H.subgraph(max(comps, key=len))
    comms = nx.community.louvain_communities(H, seed=seed)
    f = {
        "nodes": n, "edges": m, "mean_degree": deg.mean(), "degree_sd": deg.std(),
        "degree_gini": gini(deg), "degree_max": deg.max(),
        "degree_assortativity": nx.degree_assortativity_coefficient(H) if m else np.nan,
        "avg_clustering": nx.average_clustering(H), "transitivity": nx.transitivity(H),
        "components": len(comps), "lcc_fraction": lcc.number_of_nodes() / n,
        "louvain_communities": len(comms), "louvain_modularity": nx.community.modularity(H, comms),
    }
    # path lengths: exact if small, otherwise BFS from a random sample of sources
    nodes = list(lcc.nodes())
    if len(nodes) > 1:
        rng = random.Random(seed)
        src = nodes if len(nodes) <= 1000 else rng.sample(nodes, path_sources)
        tot = cnt = 0
        ecc = 0
        for s in src:
            dist = nx.single_source_shortest_path_length(lcc, s)
            tot += sum(dist.values()); cnt += len(dist) - 1
            ecc = max(ecc, max(dist.values()))
        f["avg_path_lcc"] = tot / cnt
        f["diameter_lcc" if len(src) == len(nodes) else "diameter_lcc_lowerbound"] = ecc
        f.update(lcc_spectrum(H))
    f["degree_sequence"] = deg.astype(int)
    if labels is not None:
        K = K or (max(labels.values() if isinstance(labels, dict) else labels) + 1)
        lab = labels if isinstance(labels, dict) else {i: int(l) for i, l in enumerate(labels)}
        C = mixing_counts(H, lab, K)
        f["mixing_counts"] = C
        f["mixing_matrix"] = row_normalize(C)
        f["group_sizes"] = np.bincount([lab[v] for v in H.nodes()], minlength=K)
        f["attr_assortativity"] = attribute_assortativity(C)
        parts = [{v for v in H.nodes() if lab[v] == k} for k in range(K)]
        f["attr_modularity"] = nx.community.modularity(H, [p for p in parts if p])
    return f


def scalar_features(f: dict) -> dict:
    return {k: v for k, v in f.items() if np.isscalar(v)}


def degree_ks(a, b) -> float:
    """Two-sample Kolmogorov-Smirnov distance between two degree sequences."""
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    grid = np.union1d(a, b)
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


# --------------------------------------------------------------------------- swap machinery
class SwapState:
    """
    Simple undirected graph (nodes 0..n-1) supporting degree-preserving double-edge swaps with
    INCREMENTAL triangle / average-clustering bookkeeping (no full recomputation per move).
    Average clustering follows the NetworkX convention (nodes with degree < 2 contribute 0).
    """

    def __init__(self, G):
        n = G.number_of_nodes()
        if set(G.nodes()) != set(range(n)):
            raise ValueError("nodes must be 0..n-1")
        self.n = n
        self.adj = [set() for _ in range(n)]
        self.edges = []
        for u, v in G.edges():
            if u == v:
                continue
            self.adj[u].add(v); self.adj[v].add(u)
            self.edges.append((u, v))
        self.deg = np.array([len(a) for a in self.adj])
        self.coef = [2.0 / (k * (k - 1)) if k > 1 else 0.0 for k in self.deg]
        self.tri = [0] * n
        for u, v in self.edges:
            for w in self.adj[u] & self.adj[v]:
                self.tri[w] += 1
        self.csum = sum(c * t for c, t in zip(self.coef, self.tri))

    def avg_clustering(self) -> float:
        return self.csum / self.n

    def _remove(self, u, v):
        common = self.adj[u] & self.adj[v]
        k = len(common)
        for w in common:
            self.tri[w] -= 1; self.csum -= self.coef[w]
        self.tri[u] -= k; self.csum -= self.coef[u] * k
        self.tri[v] -= k; self.csum -= self.coef[v] * k
        self.adj[u].remove(v); self.adj[v].remove(u)

    def _add(self, u, v):
        common = self.adj[u] & self.adj[v]
        k = len(common)
        for w in common:
            self.tri[w] += 1; self.csum += self.coef[w]
        self.tri[u] += k; self.csum += self.coef[u] * k
        self.tri[v] += k; self.csum += self.coef[v] * k
        self.adj[u].add(v); self.adj[v].add(u)

    def propose(self, rng):
        """Return a valid swap or None. Four distinct endpoints => no self-loop; new edges are
        checked against current adjacency => no duplicate."""
        m = len(self.edges)
        i1, i2 = rng.randrange(m), rng.randrange(m)
        if i1 == i2:
            return None
        a, b = self.edges[i1]
        c, d = self.edges[i2]
        if len({a, b, c, d}) < 4:
            return None
        n1, n2 = ((a, d), (c, b)) if rng.random() < 0.5 else ((a, c), (b, d))
        if n1[1] in self.adj[n1[0]] or n2[1] in self.adj[n2[0]]:
            return None
        return (i1, i2, (a, b), (c, d), n1, n2)

    def apply(self, mv):
        i1, i2, o1, o2, n1, n2 = mv
        self._remove(*o1); self._remove(*o2); self._add(*n1); self._add(*n2)
        self.edges[i1], self.edges[i2] = n1, n2

    def undo(self, mv):
        i1, i2, o1, o2, n1, n2 = mv
        self._remove(*n1); self._remove(*n2); self._add(*o1); self._add(*o2)
        self.edges[i1], self.edges[i2] = o1, o2

    def to_graph(self, edges=None) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(self.edges if edges is None else edges)
        return G


# --------------------------------------------------------------------------- stage 1: topology
def initial_simple_graph(degree_sequence, seed=0, mix_factor=10) -> SwapState:
    """Exact degree sequence -> simple graph (Havel-Hakimi), then randomized by degree-preserving swaps."""
    seq = [int(d) for d in degree_sequence]
    if not nx.is_graphical(seq):
        raise ValueError("degree sequence is not graphical")
    st = SwapState(nx.havel_hakimi_graph(seq))
    rng = random.Random(seed)
    for _ in range(mix_factor * len(st.edges)):
        mv = st.propose(rng)
        if mv is not None:
            st.apply(mv)
    return st


def match_clustering(state: SwapState, target, steps, rng, tol=1e-3):
    """Greedy degree-preserving swaps: accept a swap iff |C - target| does not increase."""
    cur = state.avg_clustering()
    used = 0
    for used in range(steps):
        if abs(cur - target) <= tol:
            break
        mv = state.propose(rng)
        if mv is None:
            continue
        state.apply(mv)
        new = state.avg_clustering()
        if abs(new - target) <= abs(cur - target):
            cur = new
        else:
            state.undo(mv)
    return cur, used


# --------------------------------------------------------------------------- stage 2: attributes
def _relabel(adj, labels, C, u, new):
    old = labels[u]
    if old == new:
        return
    for w in adj[u]:
        l = labels[w]
        C[old, l] -= 1; C[l, old] -= 1
        C[new, l] += 1; C[l, new] += 1
    labels[u] = new


def _cost(C, target):
    return mixing_rmse(row_normalize(C), target)


def counts_from_edges(edges, labels, K):
    C = np.zeros((K, K))
    for u, v in edges:
        C[labels[u], labels[v]] += 1
        C[labels[v], labels[u]] += 1
    return C


def fit_labels(state: SwapState, target, labels0, steps, rng, n_log=20):
    """
    Anneal a permutation of node labels (group sizes fixed) to minimise mixing RMSE on a FIXED
    topology. Temperature is calibrated from the spread of cost changes at the start
    (T0 = sd, T1 = 0.02 sd, geometric cooling), so acceptance behaviour is explicit and logged.
    Returns (best_labels, best_cost, history).
    """
    K = target.shape[0]
    labels = list(labels0)
    n = len(labels)
    C = counts_from_edges(state.edges, labels, K)
    cur = _cost(C, target)
    # calibration
    ds = []
    for _ in range(1500):
        u, v = rng.randrange(n), rng.randrange(n)
        pu, pv = labels[u], labels[v]
        if pu == pv:
            continue
        _relabel(state.adj, labels, C, u, pv); _relabel(state.adj, labels, C, v, pu)
        ds.append(_cost(C, target) - cur)
        _relabel(state.adj, labels, C, u, pu); _relabel(state.adj, labels, C, v, pv)
    sigma = float(np.std(ds))
    t0, t1 = sigma, 0.02 * sigma
    best, best_labels = cur, list(labels)
    hist, acc, tried = [], 0, 0
    for s in range(steps):
        temp = t0 * (t1 / t0) ** (s / steps)
        u, v = rng.randrange(n), rng.randrange(n)
        pu, pv = labels[u], labels[v]
        if pu != pv:
            _relabel(state.adj, labels, C, u, pv); _relabel(state.adj, labels, C, v, pu)
            new = _cost(C, target)
            d = new - cur
            tried += 1
            if d <= 0 or rng.random() < math.exp(-d / temp):
                cur = new; acc += 1
                if cur < best:
                    best, best_labels = cur, list(labels)
            else:
                _relabel(state.adj, labels, C, u, pu); _relabel(state.adj, labels, C, v, pv)
        if (s + 1) % max(steps // n_log, 1) == 0:
            hist.append({"step": s + 1, "temp": temp, "current": cur, "best": best,
                         "accept_rate": acc / max(tried, 1)})
            acc = tried = 0
    return best_labels, best, hist


# --------------------------------------------------------------------------- stage 3: constrained edge swaps
def refine_edges(state: SwapState, labels, target, eps, steps, rng, n_log=20):
    """
    Degree-preserving swaps minimising mixing RMSE subject to the HARD constraint
        |C - C0| <= eps * C0      (C0 = clustering at entry).
    Moves violating the constraint are rejected outright. Temperature calibrated as in fit_labels.
    Returns (best_edges, best_cost, history, stats). The state is left at its final (not best) edges.
    """
    K = target.shape[0]
    C = counts_from_edges(state.edges, labels, K)
    c0 = state.avg_clustering()
    lo, hi = c0 * (1 - eps), c0 * (1 + eps)
    cur = _cost(C, target)

    def delta_counts(Cm, mv, sign_old=-1, sign_new=+1):
        _, _, o1, o2, n1, n2 = mv
        for (u, v), sg in ((o1, sign_old), (o2, sign_old), (n1, sign_new), (n2, sign_new)):
            i, j = labels[u], labels[v]
            Cm[i, j] += sg; Cm[j, i] += sg

    ds = []
    for _ in range(1500):
        mv = state.propose(rng)
        if mv is None:
            continue
        Ct = C.copy(); delta_counts(Ct, mv)
        ds.append(_cost(Ct, target) - cur)
    sigma = float(np.std(ds))
    t0, t1 = sigma, 0.02 * sigma
    best, best_edges = cur, list(state.edges)
    hist, prop, feas, acc = [], 0, 0, 0
    w_prop = w_feas = w_acc = 0
    for s in range(steps):
        temp = t0 * (t1 / t0) ** (s / steps)
        mv = state.propose(rng)
        if mv is not None:
            prop += 1; w_prop += 1
            state.apply(mv)
            c = state.avg_clustering()
            if lo <= c <= hi:
                feas += 1; w_feas += 1
                Ct = C.copy(); delta_counts(Ct, mv)
                new = _cost(Ct, target)
                d = new - cur
                if d <= 0 or rng.random() < math.exp(-d / temp):
                    C, cur = Ct, new; acc += 1; w_acc += 1
                    if cur < best:
                        best, best_edges = cur, list(state.edges)
                else:
                    state.undo(mv)
            else:
                state.undo(mv)
        if (s + 1) % max(steps // n_log, 1) == 0:
            hist.append({"step": s + 1, "temp": temp, "current": cur, "best": best,
                         "clustering": state.avg_clustering(),
                         "feasible_rate": w_feas / max(w_prop, 1),
                         "accept_rate_of_feasible": w_acc / max(w_feas, 1)})
            w_prop = w_feas = w_acc = 0
    return best_edges, best, hist, {"proposed": prop, "feasible": feas, "accepted": acc, "c0": c0}


# --------------------------------------------------------------------------- stage 2b: attributes assigned per UNIT (e.g. household)
def fit_labels_grouped(state: SwapState, target, units, unit_labels0, steps, rng, n_log=20):
    """
    Like fit_labels, but labels are assigned to whole UNITS (e.g. households: every member shares one label).
    A move swaps the labels of two units of the SAME size, so the number of people per label never changes.
    Returns (node_labels, best_cost, history).
    """
    K = target.shape[0]
    n = state.n
    ulab = list(unit_labels0)
    labels = [0] * n
    for u, mem in enumerate(units):
        for v in mem:
            labels[v] = ulab[u]
    by_size = {}
    for u, mem in enumerate(units):
        by_size.setdefault(len(mem), []).append(u)
    elig = [u for u, mem in enumerate(units) if len(by_size[len(mem)]) >= 2]
    C = counts_from_edges(state.edges, labels, K)
    cur = _cost(C, target)

    def swap(u1, u2):
        l1, l2 = ulab[u1], ulab[u2]
        for v in units[u1]:
            _relabel(state.adj, labels, C, v, l2)
        for v in units[u2]:
            _relabel(state.adj, labels, C, v, l1)
        ulab[u1], ulab[u2] = l2, l1

    def propose():
        u1 = elig[rng.randrange(len(elig))]
        pool = by_size[len(units[u1])]
        u2 = pool[rng.randrange(len(pool))]
        return (u1, u2) if u1 != u2 and ulab[u1] != ulab[u2] else None

    ds = []
    for _ in range(1500):
        mv = propose()
        if mv is None:
            continue
        swap(*mv); ds.append(_cost(C, target) - cur); swap(*mv)
    sigma = float(np.std(ds)); t0, t1 = sigma, 0.02 * sigma
    best, best_labels = cur, list(labels)
    hist, acc, tried = [], 0, 0
    for s in range(steps):
        temp = t0 * (t1 / t0) ** (s / steps)
        mv = propose()
        if mv is not None:
            swap(*mv); new = _cost(C, target); d = new - cur; tried += 1
            if d <= 0 or rng.random() < math.exp(-d / temp):
                cur = new; acc += 1
                if cur < best:
                    best, best_labels = cur, list(labels)
            else:
                swap(*mv)
        if (s + 1) % max(steps // n_log, 1) == 0:
            hist.append({"step": s + 1, "temp": temp, "current": cur, "best": best, "accept_rate": acc / max(tried, 1)})
            acc = tried = 0
    return best_labels, best, hist
