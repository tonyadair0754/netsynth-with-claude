# results/

A snapshot of the outputs produced by the scripts in this repo, so reviewers can read the findings without re-running anything.
Every file here is regenerable (`python run_all.py --list`); the numbers are explained in `docs/FINDINGS.md`.

| Folder | Made by | Contents |
|---|---|---|
| `pipeline/` | `experiments/run_feature_pipeline.py` | Check 1 (what degree + clustering recover), Check 2 (income-label fitting), `run_config.json` (settings and library versions) |
| `utah_replication/` | `experiments/run_utah_replication.py` | Exp 1 replication, 1b path length vs size, 2 sensitivity, 3 income mixing by layer |
| `release_features_ut/` | `release_tools/analyze_release.py` | Feature profile of the Utah release (large `.npy` helper files are git-ignored) |
| `group_sizes/` | `release_tools/check_group_sizes_vs_inputs.py` | Generated school/daycare sizes vs input enrollment |
| `calibration/` | `experiments/calibration_demo.py` | Toy parameter-recovery grid |

Snapshot environment: Python 3.12, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, networkx 3.6.1. A re-run under networkx 3.7 / numpy 2.5.4 gave the same
Utah-release and calibration numbers and the same conclusions for the SES experiments, with small differences in later decimals (see the README, "Reproducibility").
