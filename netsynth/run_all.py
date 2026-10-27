"""
run_all.py -- guided runner. Runs the pipeline steps in a sensible order, skips steps whose inputs are missing,
and tells you what each step is for.

    python run_all.py --list                                   show the steps
    python run_all.py --steps tests,compare_demo,calibration   quick checks, no data needed (about 1 minute)
    python run_all.py --steps pipeline,replication             SES experiments (needs the Atlas CSV in data/atlas/, or --atlas PATH)
    python run_all.py --release /path/to/ut --edu /path/to/edu_site_by_state --state ut --steps analyze,reference,sizes
    python run_all.py --release /path/to/ut --edu /path/to/edu --all                       everything (about 10 minutes)

Outputs go to results/ (and data/utah_reference/ for the 'reference' step).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import paths

HERE = paths.ROOT
STEPS = {
    "tests": ("Check that the code behaves as designed (swap invariants, exchange model, generators).", None),
    "compare_demo": ("Show the observed-vs-synthetic comparison output on two made-up graphs.", None),
    "calibration": ("Toy demo: recover generator parameters from features (clustering alone vs clustering + path length).", None),
    "pipeline": ("Check 1 (what degree + clustering recover) and Check 2 (SES label fitting). Needs the Atlas CSV. ~3 min.", "atlas"),
    "replication": ("Utah experiments: replication, sensitivity, SES-by-layer. Needs the Atlas CSV; uses data/utah_reference/. ~4 min.", "atlas"),
    "analyze": ("Extract features from a state release. Needs --release (folder with <st>_population.gpkg). ~2 min.", "release"),
    "reference": ("Distil the release into data/utah_reference/. Needs --release and a previous 'analyze'.", "release"),
    "sizes": ("Compare generated school/daycare sizes with the input enrollment. Needs --release and --edu.", "edu"),
}


def run(cmd, label):
    print(f"\n{'=' * 78}\n>> {label}\n   $ {' '.join(map(str, cmd))}\n{'=' * 78}", flush=True)
    t = time.time(); r = subprocess.run([sys.executable, *map(str, cmd)], cwd=HERE)
    print(f"<< {label}: {'OK' if r.returncode == 0 else 'FAILED (exit %d)' % r.returncode} in {time.time() - t:.0f}s")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default=""); ap.add_argument("--all", action="store_true"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--atlas"); ap.add_argument("--release"); ap.add_argument("--edu"); ap.add_argument("--state", default="ut")
    a = ap.parse_args()
    if a.list or not (a.steps or a.all):
        for k, (d, _) in STEPS.items():
            print(f"  {k:13s} {d}")
        return
    wanted = list(STEPS) if a.all else [s.strip() for s in a.steps.split(",") if s.strip()]
    bad = [s for s in wanted if s not in STEPS]
    if bad:
        sys.exit(f"unknown step(s): {bad}. Use --list.")
    atlas = Path(a.atlas).resolve() if a.atlas else (paths.DEFAULT_ATLAS if paths.DEFAULT_ATLAS.exists() else None)
    given = {"atlas": atlas, "release": a.release and Path(a.release).resolve(),
             "edu": a.edu and a.release and Path(a.edu).resolve()}
    for key, p in given.items():
        if p and not p.exists():
            sys.exit(f"{key} path not found: {p}")
    ok = {}
    for s in wanted:
        need = STEPS[s][1]
        if need and not given.get(need):
            hint = f"--atlas PATH (or save the CSV as {paths.DEFAULT_ATLAS})" if need == "atlas" else f"--{need}"
            print(f"\n-- skipping '{s}': needs {hint}"); ok[s] = None; continue
        if s == "tests":
            ok[s] = run(["test_netsynth.py"], s)
        elif s == "compare_demo":
            ok[s] = run(["compare_networks.py", "--demo"], s)
        elif s == "calibration":
            ok[s] = run(["experiments/calibration_demo.py"], s)
        elif s == "pipeline":
            ok[s] = run(["experiments/run_feature_pipeline.py", given["atlas"]], s)
        elif s == "replication":
            if not paths.UTAH_REFERENCE.exists():
                print("\n-- skipping 'replication': data/utah_reference/ missing (run analyze + reference first)"); ok[s] = None; continue
            ok[s] = run(["experiments/run_utah_replication.py", given["atlas"]], s)
        elif s == "analyze":
            ok[s] = run(["release_tools/analyze_release.py", given["release"], a.state], s)
        elif s == "reference":
            ok[s] = run(["release_tools/build_utah_reference.py", given["release"], a.state], s)
        elif s == "sizes":
            ok[s] = run(["release_tools/check_group_sizes_vs_inputs.py", given["release"], given["edu"], a.state], s)
    print("\nSummary:", {k: ("ok" if v else "skipped" if v is None else "FAILED") for k, v in ok.items()})
    sys.exit(0 if all(v in (True, None) for v in ok.values()) else 1)


if __name__ == "__main__":
    main()
