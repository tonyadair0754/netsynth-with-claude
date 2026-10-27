"""Where things live. Every script imports these so it works no matter which folder you run it from."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
UTAH_REFERENCE = DATA / "utah_reference"            # small, committed (made by release_tools/build_utah_reference.py)
ATLAS_DIR = DATA / "atlas"                          # NOT committed: put the Social Capital Atlas CSV here
DEFAULT_ATLAS = ATLAS_DIR / "100_x_100_friendship_matrix.csv"
