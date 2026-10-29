# BHRF probabilities: new LC classifier vs legacy, difference per class

Same objects, same light-curve epoch: probabilities from the new feature step + LC classifier
(BHRF 2.1.0) against the ones the legacy pipeline stored. Cohort A: 1,000 objects sampled at
random from everything legacy classified. Cohort B: 100 objects whose legacy top class is
Transient.

**What drives the differences.** The model is the same on both sides, so a probability
difference is a feature difference. Those come from four inputs (a different detection set in
the two databases, a WISE match on our side only, a different period peak or a period that is
not the identical number, and fit optimisers landing elsewhere). The `same inputs` columns
repeat the quantiles on the objects where the first three are absent: same detection set, WISE
on both sides or on neither, identical period (212 objects in A, 64 in B).

On that subset the final class agrees for 97.2 % of A (93.3 % on all of A) and 89 % of B (84 %
on all of B). The disagreements that remain are near ties: the smaller of the two winning
margins is at most 0.05 in every one of them, against a median of 0.1 when the class agrees.
Which feature family decides each tie was found by swapping legacy's values for that family
into our vector until legacy's class came back:

| deciding family on the same-inputs disagreements | A (6) | B (7) |
|---|---:|---:|
| SPM fit | 1 | 3 |
| PS1 crossmatch values | 2 | 1 |
| reference-image values (distnr, chinr, sharpnr) | 0 | 2 |
| harmonic fit at the identical period | 2 | 0 |
| two families together (PS1 + reference; several small ones) | 1 | 1 |

Columns per cohort: median, 95th and 99th percentile of |p_ours − p_legacy| over all objects,
then the same three on the same-inputs objects. Final-class probabilities.

| class | A (1,000 random) median | p95 | p99 | same inputs median | p95 | p99 | B (100 transients) median | p95 | p99 | same inputs median | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SNIa | 0.000 | 0.004 | 0.015 | 0.000 | 0.004 | 0.014 | 0.004 | 0.041 | 0.103 | 0.002 | 0.034 | 0.054 |
| SESN | 0.000 | 0.003 | 0.016 | 0.000 | 0.002 | 0.021 | 0.003 | 0.032 | 0.060 | 0.002 | 0.021 | 0.027 |
| SNII | 0.000 | 0.004 | 0.023 | 0.000 | 0.003 | 0.028 | 0.004 | 0.055 | 0.104 | 0.003 | 0.028 | 0.080 |
| SNIIn | 0.000 | 0.005 | 0.024 | 0.000 | 0.003 | 0.024 | 0.004 | 0.045 | 0.074 | 0.002 | 0.035 | 0.052 |
| SLSN | 0.000 | 0.004 | 0.019 | 0.000 | 0.003 | 0.021 | 0.002 | 0.034 | 0.104 | 0.002 | 0.021 | 0.035 |
| TDE | 0.000 | 0.004 | 0.020 | 0.000 | 0.003 | 0.014 | 0.001 | 0.028 | 0.047 | 0.000 | 0.010 | 0.018 |
| QSO | 0.001 | 0.010 | 0.033 | 0.000 | 0.005 | 0.008 | 0.000 | 0.007 | 0.013 | 0.000 | 0.002 | 0.010 |
| AGN | 0.001 | 0.008 | 0.035 | 0.000 | 0.004 | 0.010 | 0.000 | 0.009 | 0.097 | 0.000 | 0.002 | 0.003 |
| Blazar | 0.001 | 0.013 | 0.043 | 0.001 | 0.006 | 0.013 | 0.000 | 0.012 | 0.077 | 0.000 | 0.004 | 0.006 |
| YSO | 0.003 | 0.028 | 0.066 | 0.001 | 0.008 | 0.020 | 0.000 | 0.016 | 0.026 | 0.000 | 0.008 | 0.038 |
| CV/Nova | 0.002 | 0.032 | 0.094 | 0.001 | 0.010 | 0.047 | 0.001 | 0.034 | 0.054 | 0.000 | 0.016 | 0.055 |
| Microlensing | 0.001 | 0.012 | 0.054 | 0.000 | 0.006 | 0.017 | 0.000 | 0.015 | 0.041 | 0.000 | 0.005 | 0.016 |
| LPV | 0.001 | 0.014 | 0.035 | 0.000 | 0.009 | 0.025 | 0.000 | 0.005 | 0.207 | 0.000 | 0.001 | 0.078 |
| CEP | 0.002 | 0.019 | 0.057 | 0.001 | 0.010 | 0.027 | 0.000 | 0.010 | 0.029 | 0.000 | 0.003 | 0.029 |
| RRLab | 0.001 | 0.014 | 0.032 | 0.000 | 0.006 | 0.017 | 0.000 | 0.006 | 0.008 | 0.000 | 0.004 | 0.008 |
| RRLc | 0.001 | 0.015 | 0.032 | 0.000 | 0.007 | 0.018 | 0.000 | 0.007 | 0.013 | 0.000 | 0.006 | 0.018 |
| DSCT | 0.001 | 0.017 | 0.039 | 0.000 | 0.005 | 0.017 | 0.000 | 0.009 | 0.019 | 0.000 | 0.002 | 0.020 |
| EA | 0.003 | 0.026 | 0.059 | 0.001 | 0.013 | 0.025 | 0.000 | 0.011 | 0.017 | 0.000 | 0.003 | 0.010 |
| EB/EW | 0.002 | 0.020 | 0.037 | 0.001 | 0.010 | 0.028 | 0.000 | 0.007 | 0.010 | 0.000 | 0.002 | 0.004 |
| RSCVn | 0.002 | 0.027 | 0.092 | 0.001 | 0.013 | 0.035 | 0.000 | 0.019 | 0.065 | 0.000 | 0.007 | 0.021 |
| Periodic-Other | 0.003 | 0.031 | 0.099 | 0.001 | 0.010 | 0.039 | 0.000 | 0.016 | 0.055 | 0.000 | 0.007 | 0.016 |
