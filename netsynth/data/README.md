# data/

| Path | In git? | What it is |
|---|---|---|
| `utah_reference/` | yes (0.4 MB) | A small summary of the Utah release of geo-synthetic-pop-usa: 50,000 whole households (ages only), group-size histograms, and the feature tables we compare against. Built by `release_tools/build_utah_reference.py`. Lets the experiments run without the 2 GB raw data. |
| `atlas/` | no | Where you save the Social Capital Atlas CSV (see `atlas/README.md`). |
| `raw/` (optional) | no | A convenient place to unzip the Utah release (`ut.zip`) and the education-site inputs if you want to re-run `release_tools/`. |

Where the raw data comes from (nothing here is real people's data; the Utah files are *synthetic*):
* Utah release and education sites: <https://osf.io/fpnc2/> (Jiang, Yin, Wang & Crooks 2024; code: <https://github.com/njiang8/geo-synthetic-pop-usa>).
* Social Capital Atlas: <https://data.humdata.org/dataset/social-capital-atlas> (Chetty et al.).
Check each source's license before redistributing anything derived from it.
