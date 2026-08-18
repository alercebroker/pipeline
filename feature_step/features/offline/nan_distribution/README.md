# NaN (NULL-value) distribution per feature — 27.5.6 vs 27.5.7a32.dev1

Percentage of NaN values per feature in `alerce.feature` for the two versions.
**NaN is stored as SQL `NULL`** (`value` is double precision; no float `'NaN'` exists),
so NaN rate = `count(value IS NULL) / count(*)` per `(name, fid)`.

`alerce.feature` is ~7.5B rows, unpartitioned, with **no index on `version`** — a
version-filtered `COUNT`/`GROUP BY` full-scans the table and hits `statement_timeout`.
Two ways around it, both here:

## Exact (full scan — the plotted numbers)
`nan_per_feature_exact.csv` — exact counts over the whole table via `exact_nan.py`
(`SET statement_timeout = 0`, one seq scan of ~775 GB / 7.5B rows, **~47 min**).
Aggregates both versions in a single pass: 27.5.6 = 934,755,458 rows, dev1 =
1,036,742,405 rows. **Mean NaN%: 47.2% (27.5.6) vs 49.4% (dev1).**

- `nan_mean_per_version.png` — mean NaN% across features per version.
- `nan_per_feature.png` — per-feature breakdown, all 209 `(name, fid)` features.
- `plot_nan.py` — regenerates both PNGs; prefers `nan_per_feature_exact.csv`,
  falls back to the sampled CSV if absent.

## Sampled (fast cross-check)
`nan_per_feature_sampled.csv` — from `TABLESAMPLE SYSTEM (0.05)` (~0.5M rows/version,
~2.2k obs/feature). Sampling runs on the physical table before `WHERE`, so it's a
uniform random subset of each version's rows → unbiased. It matched the exact scan
to within ~0.2 pt (47.1% / 49.1%) — use this when the 47-min full scan isn't worth it.

## Headline
Population NaN rates are near-tied. The versions diverge in opposite directions that
cancel: dev1 is **100% NaN on all WISE colors** (ran without xmatch enrichment) but
**~8–10 pts lower** on period/`Power_rate`/`PPE` features (full-LC → periods converge
for more objects). Everything else matches. A cohort of well-sampled objects
(`n_det ≥ 200`) understates population NaN massively (~6% vs ~47%) — do not use one.
