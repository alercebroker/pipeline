# WISE-NaN ablation on BHRF 2.1.0

**Question:** does the WISE crossmatch being absent (dev1 / no-xmatch condition:
all 11 WISE colors NaN) change BHRF predictions?

**Method (paired, controlled):** random sample of objects that DO have WISE (stored
`27.5.6` vector, `W1_W2` populated; n=3981). Predict twice from the *same* vector —
baseline (WISE present) vs ablated (the 11 WISE colors `W1_W2,W2_W3,W3_W4,g_W1..g_W4,
r_W1..r_W4` forced to NaN). Everything else identical, so the class change is purely
the WISE-NaN effect. Real BHRF 2.1.0 (SESN pickle), no recompute.

- `wise_ablation.py` — run the experiment (env: `training_py310`, has `validators`).
- `plot_wise_ablation.py` — marginal + transition plots.
- `wise_ablation.csv`, `wise_ablation_top_marginal.png`, `wise_ablation_top_transition.png`.

## Result — large effect
Blanking only WISE flips the majority of predictions:
- **top class unchanged for just 48.6%**, flat class for 39.0%.
- Top head marginal: **Periodic 87.4% → 37.2%**, **Stochastic 12.4% → 57.2%**,
  Transient 0.2% → 5.5%.
- **52.4% of baseline-Periodic objects flip to Stochastic** (1822), 5.3% to Transient;
  Stochastic is stable (92.5% stay).
- Flat: periodic classes collapse into stochastic ones — Periodic-Other→CV/Nova (586),
  LPV→YSO (442), RSCVn→YSO (264), LPV→CV/Nova (185), …

## Interpretation
BHRF relies heavily on WISE IR colors to separate Periodic (galactic variable stars)
from Stochastic (CV/Nova, YSO, AGN/QSO). With WISE gone, it systematically
over-predicts Stochastic. This is out-of-distribution: the model trained with WISE
present, so imputed-NaN WISE degrades the Periodic/Stochastic split. **Consequence:
any run without the xmatch enrichment (e.g. `27.5.7a32.dev1`, see
[[allwise-diffs-legacy-null]]) would badly corrupt these predictions** — do not use
WISE-null feature versions for classification. (Magnitude depends on the model's
NaN imputation, but dev1 genuinely has WISE=NaN, so this is dev1's real behavior.)
