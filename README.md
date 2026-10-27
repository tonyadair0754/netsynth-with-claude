# netsynth

**Build fake-but-realistic social networks from measurements of real ones, and check how realistic they actually are.**

*Status: research prototype. Nothing here has been validated against an observed social network yet; the first thing to do with it is the check described in [What's missing](#whats-missing-and-how-to-improve-it).*

---

## Overview

**What is it?** A small Python toolkit that (1) measures a network (how many friends people have, how often your friends
know each other, how people split into communities, who connects to whom by income level, ...), (2) builds a brand-new
synthetic network that has those same measurements, and (3) compares the synthetic network to a real one to see where it
falls short.

**What is it for?** We want to run "what if?" experiments, such as *what happens to inequality if we add friendships
between poorer people?*, but we can't experiment on real communities. A synthetic network lets us try interventions
safely. The catch: an experiment is only as believable as the network it runs on. So the real goal here is
**ecological validity**: synthetic networks that resemble real ones where it matters.

**Where are we?** The building blocks work and are tested. We studied how an existing open-source generator
([geo-synthetic-pop-usa](https://github.com/njiang8/geo-synthetic-pop-usa)) makes networks, reproduced its mechanism, and learned
which features it bakes in and which it misses. We have **not** yet compared anything to a real observed network.

---

## The big idea

```
   real network(s)            measured features              synthetic network             the check
 ┌─────────────────┐      ┌──────────────────────┐      ┌────────────────────┐      ┌───────────────────┐
 │ who is connected │ ──▶ │ degree, clustering,   │ ──▶ │ built to match      │ ──▶ │ compare back to    │
 │ to whom (+ SES)  │     │ communities, SES mix  │     │ those features      │     │ an OBSERVED network │
 └─────────────────┘      └──────────────────────┘      └────────────────────┘      └───────────────────┘
                              analyze / features              generate                  compare_networks.py
```

Some definitions you'll see everywhere:

| Term | Means |
|---|---|
| **Degree** | How many connections a person has. |
| **Clustering** | How often two of your friends also know each other (triangles). |
| **Communities / modularity** | Whether the network splits into tight groups; modularity near 0 = no clear groups, near 1 = very clear groups. |
| **Giant component** | The biggest island of people who are all reachable from each other. |
| **Mixing matrix** | A table of "people in group A are friends with people in group B this often" (we use income/SES deciles). |
| **Assortativity** | One number for "do like people connect to like?" (0 = no preference). |
| **Layer** | One *kind* of tie: household, workplace, school, friendship... |

### Three ways we build networks

1. **From statistics.** Start with the exact degree sequence and nudge links around (degree-preserving swaps) until clustering matches.
   Simple and controllable, but it doesn't pin down everything (see findings).
2. **From groups ("who shares a household / school / workplace").** Put people in groups with realistic sizes, then wire people
   *inside* each group with a simple rule. This is how the open-source US generator works, and we re-implemented it
   (`layered.py`) so we can change its assumptions and see what moves.
3. **Adding attributes afterwards.** Once the wiring exists, we give each person an income decile by shuffling labels
   until the table of "who befriends whom" matches the Social Capital Atlas data, **without touching the wiring**. We also
   built a separate friendship layer whose income mixing matches the Atlas by construction.

More detail (and the math, kept simple) is in [docs/METHODS.md](docs/METHODS.md).

---

## Quick start (5 minutes)

You need Python 3.10+ (tested on 3.12) and git.

```bash
git clone <this repo's URL>
cd netsynth
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_all.py --steps tests,compare_demo,calibration     # about 30 seconds, needs no data
```

You should see 16 tests pass and three steps report `OK`. Open `results/compare_demo/distributions.png` to see what a
comparison looks like (the demo compares two made-up graphs, **neither is real data**).

### Run the real experiments

The SES experiments need the Social Capital Atlas file, which we don't redistribute. Save it as
`data/atlas/100_x_100_friendship_matrix.csv` (instructions: [data/atlas/README.md](data/atlas/README.md)), then:

```bash
python run_all.py --steps pipeline,replication      # about 7 minutes; writes results/
```

`python run_all.py --list` shows every step. Each script can also be run alone, e.g.
`python experiments/run_utah_replication.py`.

### Re-derive the Utah numbers from the raw release (optional)

```bash
pip install -r requirements-release.txt
python run_all.py --release data/raw/ut --edu data/raw/edu_site_by_state --state ut --steps analyze,reference,sizes
```
(Unzip `ut.zip` from [OSF](https://osf.io/fpnc2/) into `data/raw/`. The Utah analysis ran in about 3 GB of RAM for us. Not needed for anything else.)

---

## What's in the repo

What you need in the **root** to start right after cloning, and what each piece does:

| Path | Role |
|---|---|
| `README.md`, `requirements.txt` | You are here; install list. |
| `run_all.py` | Guided runner: runs steps in order, skips what lacks inputs. Start here. |
| `netsynth.py` | Core library: measure features, degree-preserving swaps, label fitting, the wealth-exchange model. |
| `layered.py` | Group-based generator (household/work/school layers) and the friendship layer with a chosen mixing table. |
| `atlas.py` | Turns the Social Capital Atlas file into the 10x10 income-mixing target. |
| `compare_networks.py` | **The validation tool:** observed edge list vs synthetic edge list, side by side. |
| `paths.py` | Where files live, so every script works from any folder. |
| `test_netsynth.py` | 16 tests (`python -m pytest -q` or `python test_netsynth.py`). |
| `experiments/` | The experiments behind our findings (SES pipeline, Utah replication, calibration toy). |
| `release_tools/` | Scripts that read the 2 GB Utah release (optional). |
| `data/` | `utah_reference/` (small, committed) + where to put the Atlas file. |
| `results/` | A snapshot of the outputs our runs produced (small CSV/PNG; regenerable). |
| `docs/` | `METHODS.md` (how it works), `FINDINGS.md` (every number, with its source file). |
| `.gitignore`, `.gitattributes`, `.github/` | Housekeeping and automatic tests on GitHub. |

---

## Using it on your own data

**Compare a synthetic network with an observed one** (the most important thing this repo can do for you):
```bash
python compare_networks.py --observed observed_edges.txt --synthetic synthetic_edges.txt
```
Edge lists are plain text, one edge per line (`u v`, separated by spaces/tabs/commas; `#` lines are ignored). Add
`--obs-labels` / `--syn-labels` (CSV `node,label`) to compare income-mixing tables too. You get a CSV table, distribution distances, and plots. Features that differ by more than 25% are flagged `check`; that is a prompt to
look, not a verdict. Path length, diameter and component counts depend on network size, so only compare them when the two
networks are about the same size.

**Measure any network from Python:**
```python
import networkx as nx, netsynth as ns
G = nx.read_edgelist("my_edges.txt")
f = ns.extract_features(G)          # degree, clustering, communities, paths, spectrum, ... (add labels to get mixing)
```

**Build a group-based synthetic population:** see `experiments/run_utah_replication.py` for a worked example using
`layered.build_layers` (group sizes in, wired layers out).

---

## What we've found so far

Short version (every number, its source file and how much to trust it is in [docs/FINDINGS.md](docs/FINDINGS.md)).
"Measured" means a script in this repo computed it; "interpretation" means we are reading meaning into it.

1. **How the existing US generator makes networks.** *(measured)* Groups of 5 or fewer are fully connected; bigger groups
   get a small-world graph (each person has about 5 contacts in a big group, whether the group has 50 or 5,000 people).
   We confirmed this exactly on Utah's data. The paper's own validation checks demographics, not network structure.
2. **What its Utah networks look like.** *(measured)* About 4.8 contacts per person on average (max 17), clustering 0.46,
   71% of people in the giant component, 5% isolated, average path about 14 steps (long for a social network; sampled).
3. **Some inputs look off.** *(measured)* 83% of workplaces have one worker (10.5 million workplaces for 1.4 million workers).
   Daycares hold 894 children on average, while the input list says 23,180 children across all 268 daycares (10x fewer
   than assigned). Schools look reasonable at the median (532 vs 526) but have fat tails. *(interpretation)* The input site
   lists are old and not a census of children, so treat gaps as "look closer".
4. **Our code reproduces the mechanism.** *(measured)* Fed only group sizes and a household template, our generator matches
   the release within about 2% on degree, clustering, giant component and age mixing; path length is off (9.0 vs 14).
   **This check is circular** (the release is made by that very mechanism), so it says nothing about realism.
5. **Matching income mixing is easy, so it proves little.** *(measured)* The Atlas pattern is only weakly "like with like"
   (assortativity 0.07). Shuffling income labels on a fixed graph cuts the mismatch from about 0.048 to between 0.001 and 0.0035 without
   changing the wiring at all. *(interpretation)* A graph matching this table is weak evidence of realism.
6. **Degree and clustering alone don't determine a network.** *(measured)* On test networks with known answers, rebuilding
   from degree + clustering recovers a preferential-attachment network well (most features within a few percent) but badly misses community structure in a
   block-structured one (modularity 0.21 vs 0.58). Community structure needs its own target.
7. **Group-size assumptions matter, but unevenly.** *(measured, 2 seeds)* Changing workplace sizes moves mean degree
   from 4.8 to 6.7 and the giant component from 71% to 94%. Capping school/daycare sizes barely changes the measured features.
8. **Which layer a statistic describes matters.** *(measured)* The Atlas describes friendships. Forcing it onto a network where
   60% of ties are household ties (and household members share income) can't work (mismatch stays 0.15). A separate friendship layer fits
   (0.0009), but its assumed size then reshapes the whole network (average path 7.7 to 5.0, clustering 0.46 to 0.08), so the friendship
   layer's real structure has to come from data.

---

## What's missing, and how to improve it

In rough priority order:

1. **Validate against an observed network.** Run `compare_networks.py` with a real edge list. Candidates: the Facebook ego network used in
   the lab's paper (has social "circles" that can act as groups), and face-to-face contact datasets such as SocioPatterns
   (which could test the "clique if 5 or fewer, otherwise small-world" rule directly). We haven't downloaded or checked
   either; check access and licenses.
2. **Add community structure as an explicit target** (and a better optimizer). Our degree + clustering builder stalls short on high-clustering
   targets (0.438 reached vs 0.474 wanted) and ignores communities.
3. **Estimate the friendship layer from data** instead of the placeholder (average 6 friends, random wiring).
4. **Add geography/locality** and test whether it explains the long Utah path length (a hypothesis; scale alone doesn't).
5. **Features still unbuilt:** edge weights, temporal turnover, layer-specific mixing from data.
6. **Run the actual interventions.** The wealth-exchange model is implemented and tested (`netsynth.py`), but we haven't yet
   run intervention experiments on the synthetic networks; that is the end goal.
7. **More seeds and uncertainty:** most experiments use 2-3 replicates; add more and report intervals.
8. **Packaging:** make it `pip install`-able, add smoke tests for the experiment scripts, choose a license.

---

## Reproducibility

* Everything is seeded and each run writes its settings (`run_config*.json`).
* Results depend slightly on the NetworkX version. A re-run under NetworkX 3.7 / numpy 2.5.4 matched ours (3.6.1 / 2.4.4) exactly (apart from Windows line endings) for
  the Utah release analysis and the calibration toy. For the SES experiments the headline conclusions held, but some numbers differ in the later
  decimals: community-detection (Louvain) scores shift slightly, and the block-structured test network (SBM) is generated differently by the two
  versions. Compare conclusions, not the last digit.
* `results/` is a snapshot from one environment (see `results/pipeline/run_config.json`).

## Data, credits, license

* **Social Capital Atlas:** Chetty et al.; <https://data.humdata.org/dataset/social-capital-atlas>. See also <https://opportunityinsights.org/paper/social-capital-i-measurement-and-associations-with-economic-mobility/>.
* **geo-synthetic-pop-usa:** Jiang, Yin, Wang & Crooks (2024), *Scientific Data* 11:1204,
  <https://doi.org/10.1038/s41597-024-03970-1>; code <https://github.com/njiang8/geo-synthetic-pop-usa> (CC0); data <https://osf.io/fpnc2/>.
  `data/utah_reference/` is a small summary derived from their Utah release (synthetic people, not real ones).
* **License of this repository:** *not yet specified; the project owner must choose one before it is public.*
