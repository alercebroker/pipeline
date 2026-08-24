# Offline BHRF run — database statistics

Read back from `multisurvey_ztf` after the run. Companion to `BHRF_RUN_RESULTS.md`, which covers what the run computed; this covers what the database holds.

- `feature` partitions scanned: **1/32**
- `probability` partitions scanned: **1/16**
- `xmatch`: whole table (exact)

Partitions are HASH on `oid`, so a scanned partition is an unbiased pseudo-random subset of objects holding *all* of each object's rows. Percentages are therefore unbiased regardless of how many were read; absolute counts over a subset are scaled back up and marked as such.

## Table inventory

| table_name | partitions | approx_rows | total_size |
|---|---:|---:|---|
| feature | 32 | 1.416e+09 | 123 GB |
| object | 8 | 1.753e+05 | 55 GB |
| probability | 16 | 2.289e+09 | 399 GB |
| xmatch | 1 | 1.966e+07 | 2489 MB |

Row counts are the planner's `n_live_tup` estimates, not exact counts.

## Class distribution (rank-1 predictions)

`ranking = 1` is the predicted class. Classifier 6 is the top-level head (Periodic / Stochastic / Transient); 7, 8 and 9 are the conditional heads under Transient, Stochastic and Periodic; 5 is the flat 21-class classifier. **The heads run on every object, not only on the ones the top level routed to them**, so a row under classifier 7 means "if this object were a transient, it would be a TDE" -- not that it is one. Only classifier 6 and classifier 5 read as population compositions on their own.

| classifier_id | class_id | class_name | n | mean_prob | median_prob | share_pct |
|---:|---:|---|---:|---:|---:|---:|
| 5 | 9 | Periodic-Other | 456,120 | 0.2792 | 0.2644 | 37.59 |
| 5 | 3 | CV/Nova | 189,515 | 0.2069 | 0.1995 | 15.62 |
| 5 | 13 | RSCVn | 136,254 | 0.2532 | 0.2374 | 11.23 |
| 5 | 15 | SLSN | 76,132 | 0.2962 | 0.289 | 6.275 |
| 5 | 5 | EA | 73,022 | 0.2019 | 0.1904 | 6.019 |
| 5 | 20 | YSO | 64,381 | 0.2187 | 0.2025 | 5.306 |
| 5 | 2 | CEP | 51,772 | 0.253 | 0.2348 | 4.267 |
| 5 | 7 | LPV | 44,095 | 0.5295 | 0.5359 | 3.634 |
| 5 | 16 | SNII | 18,311 | 0.2243 | 0.2166 | 1.509 |
| 5 | 17 | SNIIn | 17,385 | 0.2287 | 0.2224 | 1.433 |
| 5 | 4 | DSCT | 16,385 | 0.252 | 0.2295 | 1.35 |
| 5 | 6 | EB/EW | 15,714 | 0.3139 | 0.2448 | 1.295 |
| 5 | 10 | QSO | 15,036 | 0.602 | 0.6072 | 1.239 |
| 5 | 11 | RRLab | 10,322 | 0.3429 | 0.2283 | 0.8507 |
| 5 | 8 | Microlensing | 7,324 | 0.1949 | 0.1816 | 0.6036 |
| 5 | 18 | SNIa | 6,918 | 0.2212 | 0.2001 | 0.5702 |
| 5 | 12 | RRLc | 5,105 | 0.3191 | 0.2337 | 0.4208 |
| 5 | 0 | AGN | 3,940 | 0.4341 | 0.427 | 0.3247 |
| 5 | 1 | Blazar | 2,574 | 0.3264 | 0.3406 | 0.2122 |
| 5 | 19 | TDE | 2,330 | 0.2073 | 0.1813 | 0.192 |
| 5 | 14 | SESN | 656 | 0.2598 | 0.2387 | 0.05407 |
| 6 | 0 | Periodic | 987,997 | 0.7382 | 0.752 | 81.73 |
| 6 | 2 | Transient | 139,817 | 0.7467 | 0.768 | 11.57 |
| 6 | 1 | Stochastic | 81,050 | 0.6213 | 0.528 | 6.705 |
| 7 | 5 | TDE | 657,340 | 0.3251 | 0.308 | 53.77 |
| 7 | 3 | SNIIn | 202,227 | 0.2527 | 0.244 | 16.54 |
| 7 | 1 | SLSN | 161,043 | 0.308 | 0.28 | 13.17 |
| 7 | 2 | SNII | 101,277 | 0.237 | 0.228 | 8.284 |
| 7 | 4 | SNIa | 78,629 | 0.235 | 0.23 | 6.431 |
| 7 | 0 | SESN | 22,090 | 0.2281 | 0.222 | 1.807 |
| 8 | 2 | CV/Nova | 899,260 | 0.5541 | 0.556 | 74.32 |
| 8 | 5 | YSO | 265,572 | 0.4755 | 0.454 | 21.95 |
| 8 | 3 | Microlensing | 21,871 | 0.4027 | 0.386 | 1.808 |
| 8 | 4 | QSO | 15,050 | 0.6331 | 0.64 | 1.244 |
| 8 | 0 | AGN | 4,734 | 0.4872 | 0.468 | 0.3913 |
| 8 | 1 | Blazar | 3,452 | 0.3631 | 0.352 | 0.2853 |
| 9 | 5 | Periodic-Other | 611,431 | 0.3235 | 0.308 | 50.19 |
| 9 | 8 | RSCVn | 198,222 | 0.3223 | 0.304 | 16.27 |
| 9 | 2 | EA | 158,131 | 0.2557 | 0.244 | 12.98 |
| 9 | 0 | CEP | 87,198 | 0.302 | 0.278 | 7.158 |
| 9 | 4 | LPV | 54,512 | 0.6231 | 0.638 | 4.475 |
| 9 | 1 | DSCT | 35,480 | 0.2665 | 0.246 | 2.912 |
| 9 | 7 | RRLc | 34,753 | 0.2365 | 0.216 | 2.853 |
| 9 | 6 | RRLab | 20,810 | 0.3052 | 0.228 | 1.708 |
| 9 | 3 | EB/EW | 17,718 | 0.3434 | 0.282 | 1.454 |

## AllWISE crossmatch

- link rows (`catid=0`, exact, whole table): **19,660,421**
- distinct oids matched (exact): **19,660,421**

Hit rate among the objects that have features, measured on the scanned `feature` partitions by joining their oids to `xmatch`:

- objects with features: **603,540**
- of those, matched to AllWISE: **485,688**
- **hit rate: 80.47%**

The whole-table `xmatch` count is LARGER than the number of objects with features, so a rate computed against the latter would exceed 100%. That is not an inconsistency: the crossmatch runs before the classifiability check, so objects later dropped for too few real detections keep an `xmatch` row and have no features. The rate above avoids the problem by counting only objects present in both.

Match distance (arcsec):

| min | p50 | p90 | p99 | max | mean |
|---:|---:|---:|---:|---:|---:|
| 0.0000 | 0.3051 | 0.6908 | 0.9570 | 1.0050 | 0.3540 |

## NaN rate per feature

**A NaN is a missing row, not a NULL.** `prepare_ao_features_for_db` drops NaN/inf before the writer sees them, so `value` is never NULL in this schema and `count(value IS NULL)` returns 0.00% for every feature. The rate is therefore

```
nan_pct = 100 * (1 - rows_for_this_feature / objects)
```

which is the same quantity the old wide `alerce.feature` measured as `value IS NULL` — there every (feature, band) had a row per object whether or not it had a value, so a missing row here is a NULL there. The *definitions* match; the numbers are NOT interchangeable with the ~47% in `nan_distribution/README.md`, which was measured over a different population and a different feature set. Full table in `nan_per_feature.csv`.

- objects in the scanned partitions: **603,540** (~19,313,280 over all 32)
- features x bands: **215**
- rows counted: **44,287,610**
- rows with a NULL value (must be 0): **0**
- mean NaN% across features: **65.87%**
- features >99% NaN: **0**
- features <1% NaN: **7**
- median NaN% across features: **77.95%**

(Only one overall figure is quoted because the mean across features and the row-level total are the same number by construction: mean(1 - n_f/N) == 1 - sum(n_f)/(F*N).)

### 25 highest NaN rate

| feature_id | feature_name | band | version | n | n_null | n_objects | present_pct | nan_pct |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20 | MHPS_ratio | 1 | 1 | 53,322 | 0 | 603,540 | 8.83 | 91.17 |
| 22 | MHPS_high | 1 | 1 | 53,322 | 0 | 603,540 | 8.83 | 91.17 |
| 21 | MHPS_low | 1 | 1 | 53,322 | 0 | 603,540 | 8.83 | 91.17 |
| 23 | MHPS_non_zero | 1 | 1 | 53,322 | 0 | 603,540 | 8.83 | 91.17 |
| 24 | MHPS_PN_flag | 1 | 1 | 53,322 | 0 | 603,540 | 8.83 | 91.17 |
| 27 | MHPS_high_30 | 1 | 1 | 53,330 | 0 | 603,540 | 8.84 | 91.16 |
| 26 | MHPS_low_365 | 1 | 1 | 53,330 | 0 | 603,540 | 8.84 | 91.16 |
| 25 | MHPS_ratio_365_30 | 1 | 1 | 53,330 | 0 | 603,540 | 8.84 | 91.16 |
| 20 | MHPS_ratio | 2 | 1 | 56,762 | 0 | 603,540 | 9.40 | 90.60 |
| 22 | MHPS_high | 2 | 1 | 56,762 | 0 | 603,540 | 9.40 | 90.60 |
| 23 | MHPS_non_zero | 2 | 1 | 56,762 | 0 | 603,540 | 9.40 | 90.60 |
| 24 | MHPS_PN_flag | 2 | 1 | 56,762 | 0 | 603,540 | 9.40 | 90.60 |
| 21 | MHPS_low | 2 | 1 | 56,762 | 0 | 603,540 | 9.40 | 90.60 |
| 25 | MHPS_ratio_365_30 | 2 | 1 | 56,769 | 0 | 603,540 | 9.41 | 90.59 |
| 26 | MHPS_low_365 | 2 | 1 | 56,769 | 0 | 603,540 | 9.41 | 90.59 |
| 27 | MHPS_high_30 | 2 | 1 | 56,769 | 0 | 603,540 | 9.41 | 90.59 |
| 105 | dbrightness_forced_phot_band | 2 | 1 | 59,245 | 0 | 603,540 | 9.82 | 90.18 |
| 104 | dbrightness_first_det_band | 2 | 1 | 59,245 | 0 | 603,540 | 9.82 | 90.18 |
| 104 | dbrightness_first_det_band | 1 | 1 | 60,643 | 0 | 603,540 | 10.05 | 89.95 |
| 105 | dbrightness_forced_phot_band | 1 | 1 | 60,643 | 0 | 603,540 | 10.05 | 89.95 |
| 72 | Rcs | 1 | 1 | 68,444 | 0 | 603,540 | 11.34 | 88.66 |
| 73 | Skew | 1 | 1 | 68,444 | 0 | 603,540 | 11.34 | 88.66 |
| 74 | SmallKurtosis | 1 | 1 | 68,444 | 0 | 603,540 | 11.34 | 88.66 |
| 75 | Std | 1 | 1 | 68,444 | 0 | 603,540 | 11.34 | 88.66 |
| 77 | Pvar | 1 | 1 | 68,444 | 0 | 603,540 | 11.34 | 88.66 |

## Timings

| section | seconds |
|---|---:|
| inventory | 0.1 |
| xmatch | 15.1 |
| probability | 14.8 |
| objects | 5.8 |
| feature | 2.9 |
