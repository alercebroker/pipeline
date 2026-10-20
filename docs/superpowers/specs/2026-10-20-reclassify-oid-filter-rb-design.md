# Oid filter for the squidward 2.2.0 reclassification: n_det >= 8 after the rb cut

Branch `reclass/squidward-2.2.0`, PR #641. Status: approved 2026-10-20; amended 2026-10-20 (non-goal on the docstring fix, criterion 3 wording).

## Goal

The 2.2.0 reclassification (and the 2.1.0 run whose oids it reused) classified
every object with `object.n_det >= 2`. Production classifies an object only when
at least `MIN_DETECTIONS_FEATURES` non-forced detections survive
`discard_bogus_detections` (rb < 0.55 dropped, rb NULL kept). The intended cut is 8.

The 220 rows will be deleted and the reclassification rerun from the stored
features, so `scripts/offline_reclassify_batch.py` needs an oid list that
applies the production filter. This script builds that list from the DB.

New script: `feature_step/scripts/offline_select_oids_rb.py`.

## Non-goals

- No change to the code of `offline_reclassify_batch.py` or `offline_run_batch.py`. The usage example in the docstring of the former is corrected to name `read_credentials.json` (repo rule: no credentials file without `read` in its name appears in any command).
- No feature computation, no writes to the DB, no deletion of the 220 rows (that is a separate, by-hand step as superuser).
- No filter on forced photometry: it never counts toward the minimum in production (`step.py:331`).

## Acceptance criteria

1. The script takes `--oid-file <npy>` (sorted int64 oids, the 20,620,133 of the 210 dump), `--min-detections N` (default 8), `--out <npy>`, `--credentials` (read-only json, default `R.DEFAULT_CREDENTIALS`), `--schema` (default `multisurvey`), `--workers`, `--unit-size` (default 50,000), `--minibatch` (default 500), `--max-units` (smoke), `--out-dir` (per-unit results, resume).
2. Pre-filter: one query `SELECT oid FROM <schema>.object WHERE sid = 0 AND n_det >= N` (the `select_oids` query of `offline_run_batch.py`, served by the n_det index), intersected with the input list in numpy. Raw `n_det` bounds the post-rb count from above, so no oid that would pass the rb count is dropped here.
3. rb count: for the survivors, in minibatches keyed by oid (`oid = ANY(:oids)` or an expanding `oid IN :oids`, which Postgres plans the same way and which lets the tests run the real SQL on sqlite), keep the oids for which
   `count(*)` of `detection JOIN ztf_detection` rows with `sid = 0 AND (rb IS NULL OR rb >= 0.55)` is `>= N`. This is `discard_bogus_detections` restricted to non-forced rows, verified against `lc_classifier/features/core/base.py:97-134`, plus the count at `feature_step/features/step.py:331`. Unit tests pin both: an oid with 8 rows of which one has rb 0.5 is dropped; an oid with 7 rows rb >= 0.55 plus one rb NULL is kept.
4. Units are contiguous slices of the pre-filtered list; each unit writes `<out-dir>/units/unit_NNNNN.npy` with the oids it kept, and a rerun with the same `--out-dir` skips units whose file exists (resume). A failed minibatch retries with `R._retry`; a unit that still fails is left without a file and reported.
5. `--out` is written at the end as one sorted int64 `.npy`, the concatenation of the unit files, with a `<out>.json` next to it: input count, after the n_det pre-filter, after the rb count, N, schema, elapsed.
6. Every new line carries a `#` comment; every public function has a docstring.

## Done looks like

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=<repo>/alerce_classifiers ~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check feature_step
```
exits 0, with the new tests in `feature_step/tests/unittest/test_offline_select_oids_rb.py` (DB readers mocked, no network).

Then on cpu1, the smoke: `--max-units 1` against the real DB, output count printed, before the full run.

## Calls I made (edit if wrong)

- One `--min-detections` flag drives both the n_det pre-filter and the rb count. Separate values make no sense: the pre-filter only exists to skip oids the count would drop anyway.
- Unit size 50,000 (100 minibatches of 500): ~400 units for the pre-filtered list, small enough to resume cheaply.
- Workers default to `R.default_workers()`, same as the runner; the reads are by-oid index lookups on the detection partitions, the same reads the 2.1.0 run did.
- The output list keeps the input order (ascending), so the reclassify run's units stay contiguous.
