# Findings so far

Status key: **Measured** = a script in this repo computed it (file given); **Interpretation** = our reading of it; **Hypothesis** = not tested.
Sample sizes are small (3 replicates for the SES pipeline and Utah replication, 2 for the sensitivity scenarios, 1 for the scale sweep and calibration toy), so treat differences of a few percent as noise.
The Utah release is *synthetic* (a model's output), so facts about it are facts about the generator, not about people.

## A. The existing US generator, on Utah (`results/release_features_ut/`, `results/group_sizes/`)

| # | Finding | Numbers | Source | Status |
|---|---|---|---|---|
| A1 | Within-group rule is exactly as published: groups of <=5 are cliques; bigger groups get a Newman-Watts-Strogatz graph | degree = size-1 for 100% of small groups; min degree 4 in every large group; mean degree 5.20 (school), 5.20 (daycare), 5.11 (work), 4.93 (household) | `rule_check.csv` | Measured |
| A2 | Degree in big groups does not depend on group size | mean about 5.2 in any group above 5 people | `rule_check.csv` | Measured |
| A3 | Merged network (all layers) | mean degree 4.79, max 17, degree Gini 0.40, clustering 0.461 (0.579 among people with >=2 contacts), 444,471 components, giant component 71.0%, 4.9% isolated, average path 13.8 (8 sampled sources), max distance seen 30 | `layer_features.csv` | Measured |
| A4 | Clustering comes mainly from small cliques | clustering among people with >=2 contacts: household 0.93, work 0.84, school 0.32, daycare 0.31 | `layer_features.csv` | Measured |
| A5 | Age mixing is a by-product of group composition | age assortativity: household 0.155, work 0.006 (age-blind), merged 0.379; school/daycare contain one age band by construction | `layer_features.csv`, `age_mixing_*.csv` | Measured |
| A6 | Work layer is mostly one-person workplaces | 1,097,864 workplaces for 1,429,819 workers; 83.4% have one worker; 10,515,837 workplaces *listed* (7.4 per worker); 10.4% of listed workplaces used | `group_size_summary.csv`, `release_summary.json` | Measured |
| A7 | Daycares are far larger than the input site list says | input: 23,180 enrolled across 268 sites (median 79); release: 238,690 children (10.3x), median 595, max 9,333 | `group_size_vs_inputs_ut.csv` | Measured |
| A8 | Schools match at the median, not in the tails | median 526 (input) vs 532 (release); 99th percentile 1,982 vs 2,843; total 571,148 vs 702,848 (1.23x); rank correlation of enrollment vs assigned 0.54 (daycares 0.095) | `group_size_vs_inputs_ut.csv` | Measured |
| A9 | The release's own validation covers demographics, not network structure | paper's technical validation: age/sex and household totals, survey-microdata similarity, block-group population ranks | the paper (Jiang et al. 2024), Technical Validation | Measured (by reading) |
| A10 | The input site lists are not a census of children | 2015 vintage, licensed sites | n/a | Interpretation: gaps in A7/A8 mean "look closer", not "wrong" |

## B. Our generator versus the release (`results/utah_replication/exp1*.csv`)

| # | Finding | Numbers | Status |
|---|---|---|---|
| B1 | Our layered generator, given only group-size histograms and a household template, reproduces most release features | within 2% on mean degree (4.76 vs 4.79), max degree, degree Gini, clustering (0.464 vs 0.461), giant component (0.712 vs 0.710), isolates, age assortativity (0.378 vs 0.379) | Measured |
| B2 | ...but not path length | 8.97 vs 13.84 | Measured |
| B3 | Population size alone does not explain B2 | twin path length 7.7 (25k people) rising to 9.2 (1.6M); single seed | Measured |
| B4 | Missing geography could explain B2 | not tested | **Hypothesis** |
| B5 | B1 is circular: it checks our code, not realism | | Interpretation |

## C. Sensitivity to group-size assumptions (`exp2_sensitivity.csv`, 2 seeds)

| Scenario | mean degree | clustering | giant component | avg path |
|---|---|---|---|---|
| A. Utah workplace sizes | 4.75 | 0.465 | 0.711 | 8.76 |
| B. lognormal groups, mean size 1.3 | 5.46 | 0.433 | 0.884 | 9.06 |
| C. mean size 5 | 6.28 | 0.399 | 0.933 | 7.96 |
| D. mean size 15 | 6.60 | 0.363 | 0.941 | 7.55 |
| E. mean size 50 | 6.67 | 0.341 | 0.941 | 7.66 |
| F/G. daycare cap 100 / 30, H. school cap 300 | 4.75 | 0.466 | 0.711 | 8.8-8.9 |

* Workplace size matters a lot for degree, clustering and connectivity (Measured). Scenario B has the *same average* workplace size as the release (1.3) but a different size *distribution*, and gives very different degrees: what matters is the size a typical **person** experiences, not the average per group (Interpretation).
* Capping school/daycare size barely changes these features, because degree in big groups doesn't depend on size (Measured).

## D. Income (SES) mixing (`results/pipeline/`, `results/utah_replication/exp3_ses.csv`)

| # | Finding | Numbers | Status |
|---|---|---|---|
| D1 | Shuffling income labels on a fixed graph matches the Atlas table almost perfectly, with zero change to wiring | mismatch (RMSE) from about 0.048 to 0.0009-0.0035 across 6 test graphs; 8 independent solutions all 0.0012-0.0014 | Measured |
| D2 | The Atlas table is only weakly "like with like", so D1 is easy | assortativity 0.07, modularity of income groups 0.065 | Measured |
| D3 | So matching it is weak evidence of realism | | Interpretation |
| D4 | Equally good label fits agree on the unconstrained statistics we looked at | income-group modularity 0.0650-0.0665; triangles with all-same-income 0.041-0.051 | Measured (only these two statistics; income-degree correlation untested) |
| D5 | If household members share income and the Atlas is applied to the whole merged graph, it cannot be matched | household ties are 60.4% of ties; Atlas mean diagonal 0.166; mismatch 0.164 (random) vs 0.152 (fitted) vs 0.0065 (labels free per person) | Measured |
| D6 | A separate friendship layer can match the Atlas | mismatch 0.0009 on that layer, 0.073 on the merged graph | Measured |
| D7 | The friendship layer's assumed size dominates global structure | adding it (placeholder: average 6 friends, random wiring): giant component 71% to 100%, path 7.7 to 5.0, clustering 0.46 to 0.08, mean degree 4.8 to 10.7 | Measured |

## E. What degree + clustering do and don't recover (`results/pipeline/check1_recovery_summary.csv`)
Test networks with known generators; rebuild each from degree sequence + clustering only; compare features not targeted.

| Reference | clustering target to got | modularity (ref vs rebuilt) | avg path | spectrum rho |
|---|---|---|---|---|
| BA (hubs) | 0.065 to 0.064 | 0.316 vs 0.321 | 2.94 vs 2.98 | 0.641 vs 0.659 |
| WS (small-world) | 0.474 to 0.438 (stalled) | 0.780 vs 0.701 | 4.45 vs 4.06 | 0.951 vs 0.912 |
| SBM (4 communities) | 0.108 to 0.107 | **0.582 vs 0.211** | 2.47 vs 2.34 | **0.806 vs 0.457** |

Measured: degree + clustering recover a hub network well, fall short on small-world clustering, and miss community structure badly. Interpretation: real networks with strong communities need an explicit community target.

## F. Calibration toy (`results/calibration/calibration_grid.csv`)
With a known generator (k=6, p=0.2): clustering alone leaves several nearby settings tied; clustering plus path length singles out the true one. Measured, but one toy generator with 4 replicates per setting.

## G. Not found / not done
No comparison with any observed network yet. No intervention experiments yet. No edge weights, temporal turnover or layer-specific mixing from data.
