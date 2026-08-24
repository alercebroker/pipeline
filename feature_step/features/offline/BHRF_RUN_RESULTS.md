# Offline BHRF run — results and verification

Full-catalogue offline run of the ZTF feature extraction + BHRF classification
(SERVER_QUICKSTART.md step 11), completed 2026-08-23 on `quimal-cpu1`.

**The run completed successfully: every object in the input list was processed,
no unit failed, no object errored, and every row computed on disk is present in
the database.**

All numbers below come from the 5,253 per-unit manifests in
`$RUN/bhrf_run/manifests/`, not from the end-of-run summary the script printed.
Those differ — see [Caveats](#caveats-on-the-printed-summary).

## 1. What was run

| | |
|---|---|
| Host | `quimal-cpu1` |
| Output | `/home/alerce/bhrf/bhrf_run` |
| Schema | `multisurvey_ztf` |
| Cut | `n_det >= 2` |
| Workers | 64 |
| Unit size | 5,000 oids (5,253 units) |
| Flags | `--features --load-db --no-shards` |

Input-list fingerprint (`run.json`), which pins the unit indices to a specific
oid array:

```json
{"n_oids": 26262154, "unit_size": 5000,
 "oid_sha1": "a239d8c936aab11e7b67dc2b6d04af55eaecf7b2",
 "oid_lo": 36028933559736971, "oid_hi": 36029005864334383}
```

The run was resumed at least once: 253 units (1,265,000 oids) completed in an
earlier pass, 5,000 units in the final pass.

## 2. Coverage — was every object processed?

Four independent checks, all passing.

**a. Every unit finished.** A manifest is written last, after both the shards and
the database commit, so its presence means that unit's rows are committed
(`offline_run_batch.py:475-518`). All 5,253 manifests exist, indices `0..5252`
with no gaps:

```bash
ls manifests/unit_*.json | wc -l          # 5253
for i in $(seq 0 5252); do f=$(printf "manifests/unit_%07d.json" $i); \
  [ -e "$f" ] || echo "MISSING unit $i"; done    # no output
```

**b. The manifests cover the whole input list.** Summed `n_oids` across all
units = **26,262,154** = `run.json.n_oids`. No oid was skipped.

**c. Every oid is accounted for by outcome.** 19,335,678 classified +
6,926,476 unclassifiable + 0 errors + 0 without detections = 26,262,154, exactly.

**d. The database received what was computed.** Both disk-vs-DB counters match
to the row:

| | computed | in DB |
|---|---:|---:|
| probability rows | 870,105,510 | 870,105,510 |
| feature rows | 1,416,585,465 | 1,416,585,465 |

## 3. Whole-run statistics

| | count | share |
|---|---:|---:|
| oids in input list | 26,262,154 | 100% |
| classified | 19,335,678 | 73.63% |
| unclassifiable (too few real detections) | 6,926,476 | 26.37% |
| no detections | 0 | 0% |
| errors | 0 | 0% |
| failed units | 0 | 0% |

Rows written:

| table | rows | per classified object |
|---|---:|---:|
| `probability` | 870,105,510 | 45.00 |
| `feature` | 1,416,585,465 | 73.26 |
| `xmatch` | 19,660,421 | — |

Exactly 45.00 probability rows per classified object, matching the 45-class
taxonomy with no partial writes. Feature rows vary per object (73.26 mean), as
expected — the count depends on which bands are present.

### Throughput

The final pass ran 5,000 units (24,997,154 oids) in **44.42 h** — 156 oid/s
across 64 workers, 0.559 core-s per oid. The wall clock for the 253 units of the
earlier pass was not recorded, so the total elapsed time for the run is not known.

## 4. Crossmatch (AllWISE via Xwave)

| | count | share of oids asked |
|---|---:|---:|
| oids the crossmatch was asked about | 26,262,154 | 100% |
| matched to an AllWISE counterpart | 19,660,421 | 74.86% |
| no AllWISE counterpart | 6,601,733 | 25.14% |

These reconcile exactly: `xmatch` rows written = asked − no-AllWISE
= 26,262,154 − 6,601,733 = 19,660,421, to the row. One `xmatch` row per matched
oid, no duplicates.

The 19,660,421 matched oids exceed the 19,335,678 classified ones by 324,743.
That is expected, not a double count: `n_no_allwise` is counted before the
unclassifiable check (`offline_run_batch.py:437`), so an object can match in the
cone search and then be dropped for having too few real detections.

### The 25.14% is normal

The script prints `~14% expected` next to this rate
(`offline_run_batch.py:438-439, 930`). **That expectation is wrong** and the
25.14% should not be read as an anomaly.

The 14% is the complement of the 86% recovery figure in
`WISE_NULL_CLASSIFICATION_IMPACT.md` §5, which was measured on a sample of 300
objects *already known to be WISE-null* — a rate conditional on a subpopulation
selected for being hard to match. The run's 25.14% is a marginal rate over every
object. They measure different things.

The comparable figure is in §1 of the same note: the WISE features in `27.5.6`
(the last feature version that actually carried WISE) were **~20–34% NaN**, which
is the population no-WISE rate. **25.14% falls inside that band.** No Xwave
problem is indicated.

## 5. Coverage window — what the oid list is a snapshot of

**The selection carries no date filter.** `select_oids` is

```sql
SELECT oid FROM multisurvey_ztf.object WHERE sid = :sid AND n_det >= :min_n_det
```

(`offline_run_batch.py:222`), and `offline_setup.py` materializes it once into
`features/offline/oids/run.npy` (`step_oids`, lines 231-252). Nothing rewrites
that file afterwards, and `run.json` pins its SHA-1, so the run covers exactly
"whatever satisfied `n_det >= 2` in `object` the moment that file was built" —
a snapshot, not a window. Objects that cross the threshold later are simply not
in it, and nothing in the run reports their absence.

Three facts bound what that snapshot contains. All are read from the database,
not from the run.

**The data horizon is 2026-08-14.** Over the classified objects, `lastmjd` runs
from **58270.17 (2018-06-01)** to **61266.52 (2026-08-14)**. No detection after
2026-08-14 informed any prediction here.

**The `object` table stopped receiving rows on the same date.** On
`object_part_0`, `max(created_date)` for eligible objects is **2026-08-14**, and
the daily counts fall away to nothing before it (2,915 on 08-05, 279 on 08-12,
51 on 08-13, 10 on 08-14, none after). Note `created_date` marks when the row
entered *this* table, not when the object was discovered — the oldest is
2026-06-08, which is when the table was populated, so it dates ingestion and
nothing else.

**The snapshot has not drifted.** Eligible objects (`n_det >= 2`) in
`object_part_0` number 3,281,362; scaled by the 8 hash partitions that is
**~26,250,896**, against the run's **26,262,154**. The 11,258 gap is +0.043%,
inside the imbalance between hash partitions.

So the exact build date of `run.npy` does not materially change coverage: the
source table has been static since 2026-08-14, and a list selected today would
be the same list to within a rounding error. To record the timestamp anyway, on
the run host:

```bash
stat -c '%n  modified:%y' $RUN/oids/run.npy
```

**What this means for the predictions.** They describe the ZTF catalogue as of
**2026-08-14**. They stay complete only while `object` stays static; once
ingestion resumes, objects reaching `n_det >= 2` after that date are outside
this run and nothing here will flag them. Picking them up means a fresh oid list
and a fresh `--out-dir` — the `run.json` fingerprint deliberately refuses to
resume an existing output directory against a changed list
(`offline_run_batch.py:589`), because reusing unit indices across two different
oid arrays would silently skip objects.

## 6. Conclusions

1. **The run ended well.** 5,253/5,253 units complete, 26,262,154/26,262,154
   objects processed, 0 unit failures, 0 per-oid errors.
2. **The database is complete and consistent with the computation.** Probability
   and feature row counts match disk exactly; the xmatch count reconciles with
   the crossmatch outcome to the row.
3. **73.63% of the catalogue was classified.** The remaining 26.37% are
   unclassifiable under the `n_det >= 2` cut — too few real detections after
   preprocessing — not failures.
4. **The crossmatch behaved normally**, at 74.86% AllWISE match rate.
5. **No rerun is needed.** No units are unmarked, so rerunning the step 11
   command would report `nothing to do.`
6. **Coverage is current as of 2026-08-14** and has not drifted: the eligible
   set today matches the run's oid list to within 0.043%, because `object` has
   received no rows since that date.

## Caveats on the printed summary

The end-of-run summary the script printed does **not** describe this run, and
its numbers should not be quoted. Two defects, both still present in
`offline_run_batch.py`:

**It reports the last pass, not the run.** `n_units = len(todo)` (line 839) and
the aggregate accumulates only over futures built from `todo` (line 855) — the
units *remaining* after resume. The 253 units finished earlier are invisible.
This is why the box read `units: 5,000` and `classified: 18,317,992` for a
5,253-unit, 19,335,678-object run.

**Its AllWISE denominator compounds the first defect.** Restricted to the final
pass it reported 6,508,656 / 18,317,992 = 35.5%, against a whole-run rate of
25.14%. (The printed 35.5% also predates a2c798cf2, which corrected the
denominator to include unclassifiable objects; that fix was not in the checkout
the server ran.)

Until both are fixed, read a resumed run's totals from the manifests:

```bash
jq -s '{units:length, oids:(map(.n_oids)|add), ok:(map(.n_ok)|add),
        err:(map(.n_errors)|add), unclass:(map(.n_unclassifiable)|add),
        nodet:(map(.n_no_detections)|add),
        prob:(map(.prob_rows)|add), db_prob:(map(.db_prob_rows)|add),
        feat:(map(.feat_rows)|add), db_feat:(map(.db_feat_rows)|add),
        xm:(map(.db_xmatch_rows)|add), noallwise:(map(.n_no_allwise)|add),
        asked:(map(.n_ok + .n_errors + .n_unclassifiable)|add)}' manifests/*.json
```
