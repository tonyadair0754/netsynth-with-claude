"""Invariant and hand-checkable tests. Run: python test_netsynth.py"""
import random
import networkx as nx
import numpy as np
import pandas as pd
import netsynth as ns


def test_gini():
    assert abs(ns.gini([1, 1, 1, 1])) < 1e-12
    assert abs(ns.gini([0, 0, 0, 1]) - 0.75) < 1e-12


def test_swapstate_invariants():
    G = nx.watts_strogatz_graph(120, 6, 0.1, seed=1)
    st = ns.SwapState(G)
    deg0 = st.deg.copy(); rng = random.Random(0); applied = 0
    for _ in range(8000):
        mv = st.propose(rng)
        if mv is None:
            continue
        before = (list(st.edges), st.csum)
        st.apply(mv); applied += 1
        if rng.random() < 0.3:          # undo must restore exactly
            st.undo(mv)
            assert st.edges == before[0] and abs(st.csum - before[1]) < 1e-9
    H = st.to_graph()
    assert applied > 1000
    assert [d for _, d in sorted(H.degree())] == list(deg0)           # degree sequence preserved
    assert nx.number_of_selfloops(H) == 0
    assert H.number_of_edges() == len(st.edges) == len(set(map(frozenset, st.edges)))  # no duplicates
    assert abs(st.avg_clustering() - nx.average_clustering(H)) < 1e-9   # incremental == recomputed


def test_label_counts_incremental():
    G = nx.watts_strogatz_graph(100, 6, 0.1, seed=2)
    st = ns.SwapState(G); rng = random.Random(3); K = 5
    labels = [i % K for i in range(100)]; rng.shuffle(labels)
    C = ns.counts_from_edges(st.edges, labels, K)
    for _ in range(3000):
        u, v = rng.randrange(100), rng.randrange(100)
        pu, pv = labels[u], labels[v]
        ns._relabel(st.adj, labels, C, u, pv); ns._relabel(st.adj, labels, C, v, pu)
    assert np.array_equal(C, ns.counts_from_edges(st.edges, labels, K))
    assert np.array_equal(C, ns.mixing_counts(G, dict(enumerate(labels)), K))
    assert np.array_equal(np.sort(np.bincount(labels)), np.sort(np.full(K, 20)))  # sizes fixed


def test_exchange_hand_checkable():
    # triangle 0-1-2 with pendant 3 attached to 0: degrees [3,2,2,1], not bipartite
    G = nx.Graph([(0, 1), (1, 2), (0, 2), (0, 3)])
    W = ns.weight_matrix(G)
    assert np.allclose(W.sum(axis=1), 1)
    assert np.allclose(W[0], [0, 1 / 3, 1 / 3, 1 / 3])
    r = ns.simulate(W, np.ones(4), eps=1e-12)
    assert r["converged"] and abs(r["x"].sum() - 4) < 1e-9
    assert np.allclose(r["x"], 4 * np.array([3, 2, 2, 1]) / 8, atol=1e-9)   # proportional to degree
    assert abs(ns.gini(r["x"]) - ns.gini([3, 2, 2, 1])) < 1e-9              # wealth Gini == degree Gini
    # bipartite path: non-lazy iteration oscillates, does NOT converge
    P = nx.path_graph(3)
    r2 = ns.simulate(ns.weight_matrix(P), np.array([1., 0, 0]), eps=1e-9, max_iter=500)
    assert not r2["converged"]
    assert abs(ns.lcc_spectrum(P)["lambda_min"] + 1) < 1e-9
    # disconnected (triangle + K4): each component conserves its OWN tokens, so wealth is not globally equalized
    D = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(4))
    x0 = np.array([3., 0, 0, 0, 0, 0, 0])
    r3 = ns.simulate(ns.weight_matrix(D), x0, eps=1e-12)
    assert r3["converged"]
    assert np.allclose(r3["x"][:3], 1.0) and np.allclose(r3["x"][3:], 0.0, atol=1e-9)
    assert np.allclose(r3["x"], ns.stationary_by_degree(D, x0, list(D.nodes())), atol=1e-9)


def test_directed_stationary_is_left_eigenvector():
    G = nx.DiGraph(); G.add_nodes_from(range(7))
    G.add_edges_from((i, i + 1) for i in range(5)); G.add_edges_from((i, 6) for i in range(6)); G.add_edge(6, 0)
    W = ns.weight_matrix(G)
    r = ns.simulate(W, np.ones(7), eps=1e-12, max_iter=100000)
    if r["converged"]:
        assert np.allclose(W.T @ r["x"], r["x"], atol=1e-8)


def test_degree_ks():
    assert ns.degree_ks([1, 2, 3], [1, 2, 3]) == 0
    assert ns.degree_ks([1, 1, 1], [5, 5, 5]) == 1




def test_fit_labels_preserves_topology_and_sizes():
    G = nx.watts_strogatz_graph(100, 6, 0.1, seed=4)
    st = ns.SwapState(G); K = 4
    rng = random.Random(1)
    T = np.full((K, K), 0.1) + 0.2 * np.eye(K); T /= T.sum(axis=1, keepdims=True)
    lab0 = [i % K for i in range(100)]; rng.shuffle(lab0)
    edges_before = list(st.edges)
    L, cost, _ = ns.fit_labels(st, T, lab0, 20000, rng)
    assert st.edges == edges_before                                   # topology untouched
    assert np.array_equal(np.bincount(L), np.bincount(lab0))          # group sizes fixed
    assert cost < ns._cost(ns.counts_from_edges(st.edges, lab0, K), T)
    assert abs(cost - ns.mixing_rmse(ns.row_normalize(ns.mixing_counts(G, dict(enumerate(L)), K)), T)) < 1e-12


def test_refine_edges_respects_hard_constraint():
    G = nx.watts_strogatz_graph(150, 6, 0.05, seed=5)
    st = ns.SwapState(G); K = 3; rng = random.Random(2)
    T = np.full((K, K), 0.2) + 0.4 * np.eye(K); T /= T.sum(axis=1, keepdims=True)
    lab = [i % K for i in range(150)]; rng.shuffle(lab)
    c0 = st.avg_clustering(); deg0 = st.deg.copy()
    best_edges, best, hist, stats = ns.refine_edges(st, lab, T, 0.10, 15000, rng)
    assert all(abs(h["clustering"] - c0) <= 0.10 * c0 + 1e-9 for h in hist)
    H = st.to_graph(best_edges)
    assert abs(nx.average_clustering(H) - c0) <= 0.10 * c0 + 1e-9
    assert [d for _, d in sorted(H.degree())] == list(deg0)
    assert nx.number_of_selfloops(H) == 0


def test_layered_rules():
    import layered as L
    rng = np.random.default_rng(0)
    # clique rule
    assert len(L.within_group_edges(list(range(5)), 1)) == 10 and L.within_group_edges([7], 1) == []
    # NWS rule: degree >= 4 for every member, no loops / duplicates
    e = L.within_group_edges(list(range(100, 300)), 5)
    G = nx.Graph(e); assert nx.number_of_selfloops(G) == 0 and len(e) == G.number_of_edges()
    assert min(d for _, d in G.degree()) >= 4 and 5.0 < np.mean([d for _, d in G.degree()]) < 5.5
    # partition covers everyone exactly once, sizes come from the pmf
    mem = np.arange(1000); gr = L.partition_by_size_pmf(mem, [3, 7], [0.5, 0.5], rng)
    flat = sorted(x for g in gr for x in g); assert flat == list(range(1000))
    assert all(len(g) in (3, 7) for g in gr[:-1])
    gr = L.partition_by_size_pmf(mem, [50], [1.0], rng, cap=10); assert max(len(g) for g in gr) == 10
    gl = L.partition_lognormal_weights(mem, 40, rng); assert sorted(x for g in gl for x in g) == list(range(1000))


def test_mixing_configuration_model():
    import layered as L
    rng = np.random.default_rng(3); K = 4
    Q = np.full((K, K), 0.1) + 0.6 * np.eye(K); Q /= Q.sum(1, keepdims=True)
    N = 8000; deg = rng.poisson(8, N); lab = rng.integers(0, K, N)
    E = L.mixing_configuration_model(deg, lab, Q, rng)
    assert (E[:, 0] != E[:, 1]).all() and len(E) == len(np.unique(E, axis=0))
    C = np.zeros((K, K)); np.add.at(C, (lab[E[:, 0]], lab[E[:, 1]]), 1); C = C + C.T
    P = C / C.sum(1, keepdims=True)
    assert np.sqrt(np.mean((P - Q) ** 2)) < 0.01
    realized = np.bincount(E.ravel(), minlength=N)
    assert realized.mean() > 0.97 * deg.mean() and (realized <= deg).all()


def test_fit_labels_grouped():
    G = nx.watts_strogatz_graph(120, 6, 0.1, seed=4); st = ns.SwapState(G); K = 3
    rng = random.Random(1)
    units = [list(range(i, i + 2)) for i in range(0, 60, 2)] + [list(range(i, i + 3)) for i in range(60, 120, 3)]
    T = np.full((K, K), 0.2) + 0.4 * np.eye(K); T /= T.sum(axis=1, keepdims=True)
    ul0 = [rng.randrange(K) for _ in units]
    L, cost, _ = ns.fit_labels_grouped(st, T, units, ul0, 6000, rng)
    assert all(len({L[v] for v in mem}) == 1 for mem in units)                      # members share a label
    size_lab0 = sorted((len(m), l) for m, l in zip(units, ul0)); size_lab1 = sorted((len(m), L[m[0]]) for m in units)
    assert size_lab0 == size_lab1                                                    # label counts per unit size preserved
    assert abs(cost - ns.mixing_rmse(ns.row_normalize(ns.mixing_counts(G, dict(enumerate(L)), K)), T)) < 1e-12


def test_compare_networks_identity_and_difference():
    import compare_networks as cn, tempfile, pathlib
    G = nx.watts_strogatz_graph(200, 6, 0.1, seed=1)
    tab, dist = cn.compare(G, G.copy(), outdir=pathlib.Path(tempfile.mkdtemp()))
    assert dist["degree_KS"] == 0 and dist["component_size_KS"] == 0
    assert (tab.dropna(subset=["abs_diff"]).abs_diff.abs() < 1e-9).all()
    H = nx.barabasi_albert_graph(200, 3, seed=2)
    tab2, dist2 = cn.compare(G, H, outdir=pathlib.Path(tempfile.mkdtemp()))
    assert dist2["degree_KS"] > 0.1 and (tab2.flag == "check").any()


def test_sparse_spectrum_matches_dense():
    G = nx.watts_strogatz_graph(600, 8, 0.1, seed=3)
    a, b = ns.lcc_spectrum(G, dense_limit=10 ** 9), ns.lcc_spectrum(G, dense_limit=10)
    assert all(abs(a[k] - b[k]) < 1e-6 for k in a)


def test_atlas_target_is_symmetric_doubly_stochastic():
    import atlas
    rng = np.random.default_rng(0)
    P = rng.random((10, 10)) + 2 * np.eye(10); P /= P.sum(axis=1, keepdims=True)      # fake row-stochastic mixing matrix
    Q = atlas.undirected_target(P)
    assert np.allclose(Q, Q.T, atol=1e-9) and np.allclose(Q.sum(axis=1), 1, atol=1e-6) and np.allclose(Q.sum(axis=0), 1, atol=1e-6)


def test_atlas_loader_shape():
    import atlas, tempfile, pathlib
    rng = np.random.default_rng(1)
    raw = rng.random((100, 100)); raw /= raw.sum(axis=1, keepdims=True)
    df = pd.DataFrame(raw, columns=[f"friend_prob_p{i}" for i in range(1, 101)]); df.insert(0, "own_ses_percentile", range(1, 101))
    f = pathlib.Path(tempfile.mkdtemp()) / "atlas.csv"; df.to_csv(f, index=False)
    D = atlas.load_decile_matrix(f)
    assert D.shape == (10, 10) and np.allclose(D.sum(axis=1), 1)


def test_reference_folder_is_present():
    import paths
    for name in ("household_sample.csv.gz", "size_hist_work.csv", "size_hist_school.csv", "size_hist_daycare.csv", "release_summary.json"):
        assert (paths.UTAH_REFERENCE / name).exists(), name


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
