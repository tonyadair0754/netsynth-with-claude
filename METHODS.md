# How it works (methods)

Written for someone who knows basic Python and graphs but not necessarily the math behind them. Code lives in `netsynth.py`
(general tools), `layered.py` (group-based generator) and `atlas.py` (the income-mixing target).

## 0. Vocabulary
A network is **nodes** (people) joined by **edges** (ties). *Degree* = number of edges at a node. A *triangle* = three people who all know each other;
*clustering* = how often two of a person's contacts are also tied to each other. *Modularity* measures how cleanly the network splits into groups.

## 1. Measuring a network (`netsynth.extract_features`)
Returns, for any NetworkX graph: size, mean/max degree, degree Gini (inequality of degrees), degree assortativity, average clustering and transitivity,
number/size of connected components, Louvain communities and modularity, average path length (exact for <=1000 nodes in the largest component, else sampled from 300 sources),
and the **spectrum** of the exchange matrix (below). With node group labels it also returns the group-to-group **mixing matrix** and its assortativity.

## 2. The wealth-exchange model (what the lab's paper uses)
Each node splits its wealth equally among its neighbors each step: `x(t+1) = Wᵀ x(t)`, where `W` is the adjacency matrix with every row divided by its sum
(`netsynth.weight_matrix`). Facts (all tested in `test_netsynth.py`):
* Total wealth is conserved.
* On a connected undirected graph the long-run wealth is proportional to degree, so the final wealth Gini equals the degree Gini. Changing *who* connects to whom while keeping every degree
  cannot change the long-run result; only the *speed* of getting there (and transients) can change.
* The speed is set by the second-largest eigenvalue magnitude `rho` of `W`; closer to 1 = slower. If the graph is bipartite (`lambda_min = -1`) this non-lazy iteration never settles.
* On disconnected graphs each component keeps its own total.
These are why we report `lambda2` and `rho` as features.

## 3. Stage 1: topology from degrees + clustering (`initial_simple_graph`, `match_clustering`)
1. Take the target degree sequence, build *some* simple graph with exactly those degrees (Havel-Hakimi), then shuffle it with many random **double-edge swaps**
   (take edges a-b and c-d, rewire to a-d and c-b; every node keeps its degree).
2. Greedily accept swaps that move average clustering toward the target. Triangles are updated incrementally, so each swap is cheap (`SwapState`).
Limits: it can stall short of high-clustering targets (see findings), and it says nothing about communities.

## 4. Stage 2: giving people attributes (`fit_labels`, `fit_labels_grouped`)
Keep the wiring fixed and **permute the labels** (e.g. income deciles) between people to make the mixing matrix match the target (RMSE). Because the graph never changes, every structural
feature is untouched. The search is simulated annealing; the "temperature" is calibrated from the spread of cost changes at the start (so acceptance behavior is sensible and logged).
`fit_labels_grouped` assigns one label to a whole unit (e.g. a household) and swaps same-size units, so household members share a label.

## 5. Stage 3: edge swaps under a hard cap (`refine_edges`)
Degree-preserving swaps that reduce mixing error subject to "average clustering may move by at most ε". Implemented and tested; the SES results here did not need it
(label fitting alone reached the target), so it is not used in the reported experiments.

## 6. Group-based generator (`layered.py`)
Mirrors the open-source US generator (Jiang et al. 2024):
1. **Groups.** People are put in household, school, daycare and workplace groups. Group sizes come from an empirical size histogram (`partition_by_size_pmf`) or from the original lognormal-weights mechanism (`partition_lognormal_weights`).
2. **Wiring inside a group.** Groups of 5 or fewer: everyone ties to everyone. Larger groups: a Newman-Watts-Strogatz graph (ring of 4 neighbors plus random shortcuts with probability 0.3). Consequence: a person in any big group has about 5 contacts regardless of group size.
3. **Population.** `population_from_household_template` resamples whole households from a template so household sizes and age mixes are inherited.
Layers are merged into one graph for analysis (`merge_layers`).

## 7. A friendship layer with a chosen mixing table (`mixing_configuration_model`)
Give every node a number of "stubs" (half-edges) equal to its desired friend count. For each pair of income groups, decide how many stub pairs should go between them (from the target
mixing matrix), then pair stubs at random inside that allowance. Self-ties and duplicates are dropped, so realized degrees are slightly below desired. The friendship degree used in our experiments (average 6) is a **placeholder assumption**.

## 8. The income-mixing target (`atlas.py`)
The Atlas gives, for each of 100 income percentiles, the probability that a friend falls in each percentile. We (1) average into 10x10 deciles, then (2) make it symmetric and doubly stochastic (Sinkhorn scaling) because an undirected graph needs the two directions of each group-to-group tie to agree.
Step 2 is a modeling choice, not data; it roughly assumes every income decile has the same total number of ties.

## 9. Checking a synthetic network (`compare_networks.py`)
Computes the same feature profile for both networks and prints differences, flags relative differences above a tolerance (default 25%), and reports Kolmogorov-Smirnov distances between degree distributions and component-size distributions
(0 = identical, 1 = completely different). Features that change with network size are marked `size_sensitive`.

## 10. Lessons learned while building this (so you don't repeat them)
* **A weighted penalty can silently do nothing.** An earlier version traded mixing error against clustering loss with a penalty. Measured on one seed, the annealing temperature was far larger than the per-swap score changes, so 93% of proposed swaps were accepted and the search was effectively a random walk. The "clustering collapse" it reported was just random rewiring, not a real trade-off. Always compare a penalty's scale to the noise it competes with, and prefer a hard constraint.
* **Use the right lever.** Matching a mixing table by rewiring edges destroyed clustering; shuffling labels on a fixed graph achieved a far better match for free.
* **Circular checks.** Re-creating a generator's output from features measured on that output tests your code, not realism.
* **Stats describe a layer.** Match each empirical statistic on the layer it was measured on.
