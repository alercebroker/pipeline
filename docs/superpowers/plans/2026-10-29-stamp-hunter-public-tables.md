# Stamp hunter on public tables: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the private-table mechanism so the stamp step always reads the public `taxonomy` and writes the public `probability`.

**Architecture:** One task, one commit. Delete the three `*Private` models and their test from db-plugins, drop the `model` argument from the two db helpers, drop `_db_model` and the two `DB_CONFIG.*_CLASS` reads from the step, delete the table-creation script and the two private-table test files, clean the README. A new test file pins the public tables with a config that still carries the old keys.

**Tech Stack:** Python 3.11, SQLAlchemy, pytest, db-plugins-multisurvey (editable in the rubin venv from the main `pipeline` checkout, not from this worktree).

**Spec:** `docs/superpowers/specs/2026-10-29-stamp-hunter-public-tables-design.md`

## Global Constraints

- Worktree: `/Users/panchoandrades/Desktop/repos/pipeline-hunter`, branch `feat/stamp-hunter-classifier`, PR #633. Base for the diff check: `7cf2cdd8c`.
- Interpreter: `~/.pyenv/versions/3.11.11/envs/rubin-stamp-classifier-step/bin/python`, run from `rubin_stamp_classifier_step/`.
- Every new or changed line of code carries a `#` comment; every public function and class a docstring.
- Deletion of `tests/unit/test_private_tables.py`, `tests/unit/test_create_private_tables.py` and `libs/db-plugins-multisurvey/tests/unittest/db/test_private_models.py` is authorized by the user (chat, 2026-10-29).
- Only the files named in the spec change (criterion 7). `charts/rubin_stamp_classifier_step/values.yaml` already has no private mention; leave it alone.
- Green = `~/.pyenv/versions/feature-step/bin/python /Users/panchoandrades/Desktop/repos/pipeline/.claude/scripts/test_delta.py check rubin_stamp_classifier_step` run from the worktree exits 0. Baseline at `7cf2cdd8c` holds only the two collection errors of the files being deleted, so every other failure is new.

## Review Focus

- A deployment yaml that still carries `TAXONOMY_CLASS` / `PROBABILITY_CLASS`: the step must start and use the public tables (pinned by `test_step_ignores_leftover_class_keys` below).
- `tests/unit/test_output_fields.py` and `test_sn_forwarding.py` build the step through `mock.patch` of `get_taxonomy_by_classifier_id` and `store_probability`; their call signatures lose `model=`. Pinned by the full unit run.
- `from db_plugins.db.sql.models_pipeline import Probability, Taxonomy` stays the only db_plugins import in `step.py` and `db.py`; nothing else on the branch imported the private models (`grep -rn Private rubin_stamp_classifier_step` must print nothing after the task).
- `get_taxonomy_by_classifier_id` still swallows DB errors and returns `{}`; the SQL text must still read `FROM taxonomy WHERE classifier_id = :classifier_id ORDER BY "order" ASC`.
- `store_probability` keeps `ON CONFLICT DO NOTHING` and the `(class_id, ranking)` ordering by descending probability.

---

### Task 1: public tables only

**Files:**
- Modify: `libs/db-plugins-multisurvey/db_plugins/db/sql/models_pipeline.py:1010-1080` (the block from the comment "Private copies of classifier / taxonomy / probability" through `ProbabilityPrivate.__partition_on__`, ending just before `class Feature(Base)`)
- Delete: `libs/db-plugins-multisurvey/tests/unittest/db/test_private_models.py`
- Modify: `rubin_stamp_classifier_step/rubin_stamp_classifier_step/db/db.py` (`store_probability`, `get_taxonomy_by_classifier_id`)
- Modify: `rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py:52-57,67-69,184-186,272-280`
- Delete: `rubin_stamp_classifier_step/scripts/create_private_tables.py`
- Delete: `rubin_stamp_classifier_step/tests/unit/test_create_private_tables.py`
- Delete: `rubin_stamp_classifier_step/tests/unit/test_private_tables.py`
- Modify: `rubin_stamp_classifier_step/README.md:155-161` (the seven comment lines under `SCHEMA: public`)
- Create: `rubin_stamp_classifier_step/tests/unit/test_public_tables.py`

**Interfaces:**
- Produces: `get_taxonomy_by_classifier_id(classifier_id: int, psql_connection: PSQLConnection) -> dict[str, int]` and `store_probability(psql_connection, classifier_id: int, classifier_version: str, class_taxonomy: dict[str, int], predictions: list[dict]) -> None` in `db.py`, both without `model`.

- [ ] **Step 1: Delete the three test files the user authorized**

```bash
git rm rubin_stamp_classifier_step/tests/unit/test_private_tables.py rubin_stamp_classifier_step/tests/unit/test_create_private_tables.py libs/db-plugins-multisurvey/tests/unittest/db/test_private_models.py
```

- [ ] **Step 2: Write the failing test `tests/unit/test_public_tables.py`**

Reuse the `RecordingConnection`, `executed_sql`, `prediction` and `build_step` helpers from the deleted `test_private_tables.py` (copy them in; the file is gone). `LEFTOVER = {"TAXONOMY_CLASS": "nowhere.Bogus", "PROBABILITY_CLASS": "nowhere.Bogus"}`.

```python
def test_taxonomy_is_read_from_the_public_table():
    connection = RecordingConnection(rows=[{"class_id": 0, "class_name": "candidate"}])
    mapping = get_taxonomy_by_classifier_id(10, connection)
    assert "FROM taxonomy WHERE" in executed_sql(connection)
    assert mapping == {"candidate": 0}

def test_probabilities_are_written_to_the_public_table():
    connection = RecordingConnection()
    store_probability(connection, 10, "1.0.0", {"candidate": 1, "not_candidate": 0}, [prediction()])
    sql = executed_sql(connection)
    assert sql.startswith("INSERT INTO probability (")
    assert "ON CONFLICT DO NOTHING" in sql
    _, rows = connection.executed[0]
    assert [(r["class_id"], r["ranking"]) for r in rows] == [(1, 1), (0, 2)]

def test_step_ignores_leftover_class_keys():
    step, read_taxonomy = build_step(LEFTOVER)          # must not raise: the keys are not read
    assert "model" not in read_taxonomy.call_args.kwargs
    with mock.patch("rubin_stamp_classifier_step.step.store_probability") as store:
        step.post_execute(prediction_messages())
    assert "model" not in store.call_args.kwargs

def test_db_helpers_take_no_model_argument():
    assert "model" not in inspect.signature(get_taxonomy_by_classifier_id).parameters
    assert "model" not in inspect.signature(store_probability).parameters
```

- [ ] **Step 3: Run it to see it fail**

Run: `~/.pyenv/versions/3.11.11/envs/rubin-stamp-classifier-step/bin/python -m pytest tests/unit/test_public_tables.py -q`
Expected: `test_step_ignores_leftover_class_keys` fails (`get_class("nowhere.Bogus")` raises) and `test_db_helpers_take_no_model_argument` fails; the other two pass.

- [ ] **Step 4: Remove the private models from db-plugins**

Delete `models_pipeline.py` from the "Private copies" comment block through `ProbabilityPrivate.__partition_on__` (just before `class Feature(Base)`). No other line in that file changes.

- [ ] **Step 5: Drop `model` from the two db helpers in `db.py`**

`store_probability(psql_connection, classifier_id, classifier_version, class_taxonomy, predictions) -> None`: `insert(Probability)`, docstring says it inserts into the public probability table. `get_taxonomy_by_classifier_id(classifier_id, psql_connection) -> dict[str, int]`: `table = Taxonomy.__tablename__`, docstring without the "private copy" sentence. Behaviour otherwise unchanged (Review Focus lines 4 and 5).

- [ ] **Step 6: Drop the model selection from `step.py`**

Remove `_db_model`, `self.taxonomy_model`, `self.probability_model` and their comment; call `get_taxonomy_by_classifier_id(self.classifier_id, self.psql_connection)` and `store_probability(...)` without `model=`. Remove the `from db_plugins.db.sql.models_pipeline import Probability, Taxonomy` line from `step.py` only if nothing else in the file uses the names (check with grep).

- [ ] **Step 7: Delete the script and clean the README**

`git rm rubin_stamp_classifier_step/scripts/create_private_tables.py`. In the README, delete the seven comment lines from `# Optional. A classifier that is not public yet` through `# The tables come from the models: scripts/create_private_tables.py.` Nothing else in the README changes.

- [ ] **Step 8: Run the spec checks**

```bash
grep -rn -i private libs/db-plugins-multisurvey
grep -n -i -E 'private|TAXONOMY_CLASS|PROBABILITY_CLASS' rubin_stamp_classifier_step/README.md charts/rubin_stamp_classifier_step/values.yaml
grep -rn Private rubin_stamp_classifier_step
git diff 7cf2cdd8c..HEAD --stat
```
Expected: the three greps print nothing; the stat lists only the files in this task plus the spec and this plan.

- [ ] **Step 9: Run the suite against the baseline**

Run from the worktree root: `~/.pyenv/versions/feature-step/bin/python /Users/panchoandrades/Desktop/repos/pipeline/.claude/scripts/test_delta.py check rubin_stamp_classifier_step`
Expected: exit 0, `new failures: 0`, `fixed: 2`.

- [ ] **Step 10: Commit**

```bash
git add -A libs/db-plugins-multisurvey/db_plugins/db/sql/models_pipeline.py rubin_stamp_classifier_step/rubin_stamp_classifier_step rubin_stamp_classifier_step/README.md rubin_stamp_classifier_step/tests/unit/test_public_tables.py
git commit -m "refactor(rubin_stamp_classifier_step): public taxonomy and probability tables only"
```
(the `git rm` of Steps 1 and 7 is already staged.)
