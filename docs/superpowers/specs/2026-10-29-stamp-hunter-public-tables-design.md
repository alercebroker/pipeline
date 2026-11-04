# Stamp hunter classifier: public tables only

Date: 2026-10-29. Branch `feat/stamp-hunter-classifier`, draft PR #633 to `multisurvey`.
Supersedes the private-table sections of
[2026-09-16-stamp-hunter-classifier-status.md](2026-09-16-stamp-hunter-classifier-status.md).

## Goal

Decision 2026-10-29: the hunter deployment uses the regular tables, like
rubin. The step reads its classes from the public `taxonomy` (db_plugins
`Taxonomy`) and writes rows to the public `probability` (db_plugins
`Probability`), unconditionally, directly to the DB, no scribe. The private
copies and the code that selects them go away. The step never writes
`classifier` or `taxonomy`; their hunter rows are inserted by hand.

## Non-goals

- Inserting the hunter row into the public `classifier` / `taxonomy` tables,
  or dropping `*_private` tables on any database. SQL on the live DB is the
  user's.
- The rest of the PR #633 review checklist (Dockerfile `ARG model` scope,
  CI unit job, mapper crop order, error logs, README hunter block).
- Integration tests, deployment yaml, `sn_candidates` topic and ACLs.
- `local_config_hunter.yaml` (git-ignored, the user's).

## Acceptance criteria

1. `libs/db-plugins-multisurvey/db_plugins/db/sql/models_pipeline.py` has no
   `ClassifierPrivate`, `TaxonomyPrivate` or `ProbabilityPrivate`, and
   `libs/db-plugins-multisurvey/tests/unittest/db/test_private_models.py`
   is gone. Check: `grep -rn -i private libs/db-plugins-multisurvey` prints nothing.
2. `rubin_stamp_classifier_step/rubin_stamp_classifier_step/db/db.py`:
   `get_taxonomy_by_classifier_id` and `store_probability` take no `model`
   argument. The taxonomy query reads `Taxonomy.__tablename__` and the insert
   targets `Probability`, both imported from db_plugins.
3. `rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py` has no
   `_db_model`, `taxonomy_model` or `probability_model`; `DB_CONFIG` keys
   `TAXONOMY_CLASS` / `PROBABILITY_CLASS` are not read. A config that still
   carries them changes nothing (ignored, no error).
4. `rubin_stamp_classifier_step/scripts/create_private_tables.py`,
   `rubin_stamp_classifier_step/tests/unit/test_create_private_tables.py` and
   `rubin_stamp_classifier_step/tests/unit/test_private_tables.py` are gone.
5. `rubin_stamp_classifier_step/README.md` and
   `charts/rubin_stamp_classifier_step/values.yaml` mention no private
   tables, `*_CLASS` DB keys or `create_private_tables`. Check:
   `grep -n -i -E 'private|TAXONOMY_CLASS|PROBABILITY_CLASS' rubin_stamp_classifier_step/README.md charts/rubin_stamp_classifier_step/values.yaml`
   prints nothing.
6. A new unit test, `rubin_stamp_classifier_step/tests/unit/test_public_tables.py`,
   with no database: the step built with the stub model and a `DB_CONFIG`
   that carries `TAXONOMY_CLASS` / `PROBABILITY_CLASS` pointing at a bogus
   class still queries `FROM taxonomy` and inserts into `probability`.
   Covers criteria 2 and 3.
7. Nothing else on the branch changes: forwarder, model loading,
   `MODEL_VERSION`, `RENAME_STAMP_COLUMNS`, output trimming, Dockerfile
   groups stay as they are. Check: `git diff 7cf2cdd8c..HEAD --stat` lists
   only the files named above plus this spec.

## Done looks like

```
~/.pyenv/versions/3.11.11/envs/rubin-stamp-classifier-step/bin/python -m pytest tests/unit -q
```
from `rubin_stamp_classifier_step/`, green except failures already in
`.claude/test-baselines/rubin_stamp_classifier_step.txt` (recorded on this
branch at 7cf2cdd8c before the change, since the step has no unit tests on
`multisurvey`). The db-plugins unit tests stay in the state they had at
7cf2cdd8c.

## Calls made here (check these)

- Remove the whole table-selection mechanism, not only the `*_CLASS` keys:
  going back to the earlier `*_TABLE` keys (commit c593d5455) would keep a
  switch nobody uses. Rubin behaviour before the hunter work is the target.
- One commit on the existing branch, not a revert chain: the four commits
  (c00c57a12, 5adfb326d, 7cf2cdd8c, 4e0e16522) interleave with later fixes
  (b4bec6491, f026c5b5b), so `git revert` would conflict and read worse.
- Deleting three committed test files is the user's edit (test files are
  read-only for the agent). The implementer removes them only after the user
  says so in chat; until then criteria 1 and 4 are blocked.
- Spec and work go on `feat/stamp-hunter-classifier` in the
  `pipeline-hunter` worktree and post on PR #633; no new branch, since that
  PR is the one address for this feature's review.
