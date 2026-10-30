# Stamp step: forward only first stamps, hunter classes nuclear / off-nuclear

Date: 2026-10-30. Branch `feat/stamp-hunter-classifier`, draft PR #633 to
`multisurvey`. The hunter part goes on `feat/hunter-crop-then-stack` in
`alerce_classifiers` (draft PR #94); the pipeline PR bumps the submodule.
The code was written and run against production on the night of 2026-10-29
(252,521 alerts, 2,637 SN forwarded, 2,592 hunter rows); this spec puts it
through the build flow.

## Goal

The rubin deployment forwards an SN candidate to the hunter only when the
alert is the object's first stamp, which is exactly when its own upsert into
`probability` writes a row. The upsert reports what it wrote (`RETURNING oid`)
and the forwarder keeps only those oids. The hunter's classes are named as
the `taxonomy` rows of classifier 10 (`nuclear_classifier`): `nuclear` and
`off-nuclear`.

## Non-goals

- Deduplicating oids before the model runs (`_messages_to_dto` already does;
  `execute` still emits every message). Only the DB write deduplicates.
- The hunter's own first-stamp rule: it inherits the same `store_probability`
  and gets it for free, no hunter-specific code.
- SQL on the live DB (classifier 10, taxonomy rows, deleted rows): done by
  hand on 2026-10-30, not in the repo.
- `local_config_*.yaml`, consumer-group offsets, `out/`, `parity_tools.py`:
  local, never committed.
- Building and publishing the image: an operate step after this PR's gate.

## Acceptance criteria

1. `rubin_stamp_classifier_step/rubin_stamp_classifier_step/db/db.py`:
   `store_probability` returns `set[int]`, the oids of the rows the upsert
   inserted or updated. The statement carries `RETURNING probability.oid`
   and its guard is `probability.lastmjd >= excluded.lastmjd` (a replay of
   the same first alert counts as written). Empty input returns `set()`.
   Test: `test_public_tables.py::test_store_probability_returns_the_written_oids`.
2. Same function: when a batch carries two alerts of one oid, only the one
   with the smallest `midpointMjdTai` reaches the statement (one row per
   `(oid, class)` per statement; two would raise CardinalityViolation, seen
   2026-10-30). Test:
   `test_public_tables.py::test_store_probability_keeps_the_oldest_alert_per_oid`.
3. `rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py`:
   `forward_sn_candidates(predictions, written)` forwards a ranking-1 SN
   alert only when its oid is in `written`; the batch log line reads
   `Forwarded F of N alerts as SN candidates (K SN alerts skipped: not the
   object's first stamp)`. `post_execute` passes the set `store_probability`
   returned. Test: `test_sn_forwarding.py::test_only_first_stamps_are_forwarded`.
4. `written=None` forwards every ranking-1 SN alert (behaviour before this
   change, for callers without a DB answer). Test:
   `test_sn_forwarding.py::test_without_db_answer_every_sn_is_forwarded`.
5. No local bypass in the committed code. Check:
   `grep -n -E 'STAMP_SKIP_DB_WRITE|^import os' rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py`
   prints nothing. `post_execute` always calls `store_probability`.
6. `alerce_classifiers/alerce_classifiers/hunter/model.py`:
   `CLASSES = ("nuclear", "off-nuclear")`, softmax index 1 (`off-nuclear`)
   stays the positive class the training set called `candidate`. Tests
   `tests/unit/test_hunter_model.py`, `test_hunter_mapper.py`,
   `test_hunter_skip.py` name the new classes. The pipeline branch points the
   submodule at that commit.
7. Nothing else changes. Check: `git diff f521b2727..HEAD --stat` lists only
   `db/db.py`, `step.py`, the four unit test files edited on 2026-10-30
   (`test_model_version.py`, `test_public_tables.py`, `test_skipped_alerts.py`,
   `test_sn_forwarding.py`), the submodule pointer, this spec and its plan.

## Done looks like

```
cd rubin_stamp_classifier_step && PYTHONPATH=..:../libs/db-plugins-multisurvey:../alerce_classifiers \
  ~/.pyenv/versions/3.11.11/envs/rubin-stamp-classifier-step/bin/python -m pytest tests/unit -q
```
from the `pipeline-hunter` worktree: no failure outside
`.claude/test-baselines/rubin_stamp_classifier_step.txt` (60 passed on
2026-10-30). And from `alerce_classifiers/`:
```
~/.pyenv/versions/3.11.11/envs/rubin-stamp-classifier-step/bin/python -m pytest \
  tests/unit/test_hunter_model.py tests/unit/test_hunter_mapper.py tests/unit/test_hunter_skip.py -q
```
17 passed, 1 skipped, as on 2026-10-30 (the whole `tests/unit` does not collect in
the rubin venv: `test_squidward_model.py` needs `imblearn`).

## Calls made here (check these)

- `>=` instead of `>`: the production scribe guard is `<` (strictly newer
  wins) and rubin's was `>` (strictly older wins). With `>=` a replay of the
  first alert rewrites the same values and the forwarder re-emits the SN, so
  the hunter can be re-fed by replaying. Cost: one no-op update per replayed
  first stamp.
- Dedup keeps the oldest alert per oid, the same rule the guard applies to
  stored rows, so batch order never changes the result.
- The four committed test files were edited on 2026-10-30 in chat with the
  user (the `written=` argument of the recording connection, the
  `store_probability` side effect). They stay as edited; no new test file.
- Hunter rename goes as one more commit on `feat/hunter-crop-then-stack`
  (PR #94), the open hunter branch, not a third stacked branch.
- The `# LOCAL ONLY` bypass is removed in the same commit; `commit_guard`
  blocks it anyway.
