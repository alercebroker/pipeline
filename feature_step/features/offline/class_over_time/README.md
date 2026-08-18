# BHRF 2.1.0 class distribution over time (stored predictions)

Class composition of the stored BHRF predictions, bucketed by **`lastmjd`
(last detection)**. BHRF re-runs on each new detection, so `lastmjd` ≈ when the
stored prediction was last (re)generated — the best available proxy, since
`alerce.probability` has **no processing-date or feature-version column** (it
holds one current prediction per object). Pure DB, no model.

- `fetch_class_dates.py` — random sample: `TABLESAMPLE` the BHRF flat and top-head
  probability partitions (`ranking = 1`) for oid+class, then join `alerce.object`
  by oid for `firstmjd`/`lastmjd`. Writes `class_dates_{top,flat}.csv`.
- `plot_class_dates.py` — 100%-stacked composition (top) + absolute counts (bottom),
  monthly bins by `lastmjd`. Switch `DATECOL`/`FREQ` at the top for other axes.
- `class_dates_top_lastmjd.png` — top 3-way head (Periodic/Stochastic/Transient).
- `class_dates_flat_lastmjd.png` — flat leaf, top 8 classes + Other.

## What the Oct–Nov 2025 step actually is — WISE-loss from the feature-version rollover
The top head steps from ~71% Periodic (lastmjd ≤ 2025-09) to ~37% Periodic
(lastmjd ≥ 2025-12), Stochastic taking over. This is **not** mainly a detection-cadence
selection effect (an earlier hypothesis). It is the **27.5.6 → WISE-null feature
rollover** corrupting the Periodic/Stochastic split:

- BHRF re-runs on each new detection using the *then-current* feature version, so
  `lastmjd` ≈ when the object was last classified. The current ZTF pipeline runs
  **without xmatch** (`multisurvey_ztf.xmatch` empty) → recomputed features have
  **WISE = NaN**. The rollover date (~2025-10, see the version timeline in
  OFFLINE_VS_LEGACY_VALIDATION.md §1) matches the step.
- The WISE ablation (`../wise_ablation/`) shows WISE-NaN flips Periodic→Stochastic by
  exactly this magnitude — and the **recent-era stored dist (36.8% Periodic) ≈ the
  ablated model output (37.2% Periodic)**. See [[wise-nan-breaks-bhrf-periodic]].

So the **recent months are WISE-null (dev1-condition) predictions that over-call
Stochastic**, not the "trustworthy current mix"; the old-era tail (27.5.6, WISE
present) is the more physically correct split. Loose end: ablation predicts ~5.5%
Transient vs 1.8% stored — real dev1 also has better-populated period features that
suppress transient calls. Decisive check = version attribution by reconstruction.
`firstmjd` is in the CSVs as an alternate axis but is unrelated to classification time.
