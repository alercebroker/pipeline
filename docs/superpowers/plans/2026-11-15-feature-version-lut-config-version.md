# Config version in feature_version_lut: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the row of `feature_version_lut` a feature step uses is chosen by (package version, `FEATURE_VERSION` from the config, `sid`, `tid`), and created when absent.

**Architecture:** one new column on the `FeatureVersionLut` model, one new positional argument on `get_or_create_version_id`, one config read in `FeatureStep.__init__`. The id still reaches the scribe as `features_version`; nothing downstream changes.

**Tech Stack:** Python 3.10, SQLAlchemy `text()` queries, `unittest` + `mock`, run with `~/.pyenv/versions/feature-step/bin/python`.

**Spec:** `docs/superpowers/specs/2026-11-15-feature-version-lut-config-version-design.md`

**Branch / PR:** `refactor/feature-step-surveys`, draft PR #646 (Francisco's decision: this change rides on that PR).

## Global Constraints

- Config key `FEATURE_VERSION`, top level. Column `feature_version`, `VARCHAR`, not nullable, server default `'1.0.0'`. Primary key of the table unchanged.
- `version_id` rule unchanged: `MAX(version_id) + 1` over the (`sid`, `tid`) of the survey.
- No change to the scribe, the `feature` table, `charts/`, `feature_step/scripts/local_sim`, or how `version_name` is obtained. No `ALTER TABLE` is run anywhere.
- Every new or changed line carries a `#` comment; every public function and class a docstring (a `def`/`class` line with a docstring below needs no `#`).
- Tests go in a new file. Committed test files are read-only except the four lines listed in Task 1, Step 5, which need Francisco's authorization before the step runs.
- Never commit `poetry.lock`. Test command prefix: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=<repo>/alerce_classifiers`.
- Green means `~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check feature_step` exits 0 (baseline: 8 failures).

## Review Focus

1. The table has no `feature_version` column yet (image deployed before the `ALTER TABLE`): the step must fail at start with the database error, not write under id 1. Pinned by `test_select_error_is_raised`.
2. `FEATURE_VERSION` written unquoted in yaml arrives as a float or int (`1.0`, `2`): it is used as `str(value)`, so `1.0` looks up `"1.0"`. Pinned by `test_non_string_version_is_used_as_text`.
3. `FEATURE_VERSION` with surrounding spaces (`" 0.1.0 "`): stripped before the lookup, so it does not create a second row. Pinned by `test_version_is_stripped`.
4. `FEATURE_VERSION` of only spaces: treated as empty, raises. Pinned by `test_blank_version_raises`.
5. The insert succeeds but is not committed: the next pod would not see the row. Pinned by the `commits == 1` assertion of `test_new_combination_inserts_next_id`.

---

### Task 1: `feature_version` in the model, the lookup and the step (criteria 1-8)

**Files:**
- Create: `feature_step/tests/unittest/test_feature_version_lut.py`
- Modify: `libs/db-plugins-multisurvey/db_plugins/db/sql/models_pipeline.py` (`FeatureVersionLut`, ~line 1104)
- Modify: `feature_step/features/database.py` (`get_or_create_version_id`, ~line 117)
- Modify: `feature_step/features/step.py` (~lines 98-102)
- Modify, one line each, with authorization: `feature_step/tests/unittest/test_surveys_golden.py` (`build_step` config), `feature_step/tests/unittest/test_surveys_step.py` (`config_for` and the `assert_called_once_with` of `test_version_and_lut_are_resolved_with_the_survey_sid_and_tid`), `feature_step/tests/unittest/test_step_ztf_multisurvey.py` (`build_step` config)

**Interfaces:**
- Produces: `get_or_create_version_id(db_sql, schema: str, version_name: str, feature_version: str, sid: int, tid: int, logger=None) -> int`
- Produces: `FeatureStep.feature_version: str`, the stripped text of `config["FEATURE_VERSION"]`
- Produces: `FeatureVersionLut.feature_version = Column(VARCHAR, nullable=False, server_default="1.0.0")`

- [ ] **Step 1: Write the failing tests in `test_feature_version_lut.py`**

A `FakeDb` whose `session()` is a context manager yielding a `FakeSession`. `FakeSession.execute(query, params)` appends `(str(query), params)` to `calls` and returns the next scripted result (an object with `fetchone()`); `commit()` increments `commits`. A scripted result may be an exception instance, which `execute` raises.

```python
V = ("27.8.0", "0.1.0", 1, 1)  # version_name, feature_version, sid, tid

class ModelTestCase:  # criterion 1
    def test_model_has_feature_version_column(self):
        col = FeatureVersionLut.__table__.c.feature_version
        assert isinstance(col.type, VARCHAR) and col.nullable is False
        assert col.server_default.arg == "1.0.0"
        assert [c.name for c in FeatureVersionLut.__table__.primary_key] == ["version_id", "sid"]

class LookupTestCase:
    def test_existing_combination_returns_its_id(self):  # criterion 2
        db = FakeDb([row(7)])
        assert get_or_create_version_id(db, "multisurvey", *V) == 7
        assert len(db.calls) == 1 and db.commits == 0  # one select, no insert

    def test_select_filters_by_the_four_values(self):  # criterion 4
        db = FakeDb([row(7)])
        get_or_create_version_id(db, "multisurvey", *V)
        sql, params = db.calls[0]
        for cond in ("version_name = :version_name", "feature_version = :feature_version", "sid = :sid", "tid = :tid"):
            assert cond in sql
        assert params == {"version_name": "27.8.0", "feature_version": "0.1.0", "sid": 1, "tid": 1}

    def test_new_combination_inserts_next_id(self):  # criterion 3
        db = FakeDb([row(None, found=False), row(4), None])  # select misses, max is 4, insert
        assert get_or_create_version_id(db, "multisurvey", *V) == 5
        sql, params = db.calls[2]
        assert "INSERT INTO multisurvey.feature_version_lut" in sql and "feature_version" in sql
        assert params == {"version_id": 5, "version_name": "27.8.0", "feature_version": "0.1.0", "sid": 1, "tid": 1}
        assert db.calls[1][1] == {"sid": 1, "tid": 1}  # max is taken over sid and tid only
        assert db.commits == 1

    def test_select_error_is_raised(self):  # criterion 7
        db = FakeDb([RuntimeError("column feature_version does not exist")])
        with pytest.raises(RuntimeError): get_or_create_version_id(db, "multisurvey", *V)

    def test_insert_error_is_raised(self):  # criterion 7
        db = FakeDb([row(None, found=False), row(4), RuntimeError("duplicate key")])
        with pytest.raises(RuntimeError): get_or_create_version_id(db, "multisurvey", *V)

    def test_no_connection_returns_none(self):  # criterion 7
        assert get_or_create_version_id(None, "multisurvey", *V) is None

class StepTestCase:  # config = the three mocked Kafka configs of test_surveys_golden + SURVEY "lsst" + overrides
    def test_step_passes_config_version(self):  # criterion 5
        # patch step_module.get_or_create_version_id and get_feature_name_lut
        version_id.assert_called_once_with(db_sql, "multisurvey", version("feature-step"), "0.1.0", 1, 1, step.logger)
        assert step.feature_version == "0.1.0"

    def test_missing_version_raises(self):  # criterion 6
        # no FEATURE_VERSION key: ValueError whose text contains "FEATURE_VERSION"; version_id mock not called
    def test_empty_version_raises(self):    # FEATURE_VERSION "" : same
    def test_blank_version_raises(self):    # FEATURE_VERSION "   " : same
    def test_version_is_stripped(self):     # " 0.1.0 " -> lookup called with "0.1.0"
    def test_non_string_version_is_used_as_text(self):  # 1.0 -> lookup called with "1.0"
```

Written as `unittest.TestCase` classes like the neighbouring files. If `db_plugins.__file__` does not resolve under `libs/db-plugins-multisurvey` in the feature-step env, stop and report instead of working around it.

- [ ] **Step 2: Run, expect FAIL**

Run: `<prefix> ~/.pyenv/versions/feature-step/bin/python -m pytest tests/unittest/test_feature_version_lut.py -q` from `feature_step/`.
Expected: every test fails except `test_no_connection_returns_none` (`TypeError` on the extra argument, `AttributeError` on the column, no `ValueError`). Note which fail and why in the report.

- [ ] **Step 3: Commit the test file alone**

`test(feature_step): feature_version_lut rows are chosen by package version and FEATURE_VERSION`

- [ ] **Step 4: Implement**

- `models_pipeline.py`: add the column from Interfaces after `version_name`.
- `database.py`: new signature from Interfaces; the select and the insert carry `feature_version`; the `try/except` that returns `1` is removed so exceptions propagate (keep the `db_sql is None` branch and the info logs, which now also name `feature_version`).
- `step.py`: before the version lookup, read `config.get("FEATURE_VERSION")`; `None` raises `ValueError("FEATURE_VERSION is missing from the step config")`; otherwise `str(value).strip()`, and an empty result raises the same error with "is empty"; store as `self.feature_version` and pass it as the fourth argument.

- [ ] **Step 5: The four authorized lines in committed tests**

Add `"FEATURE_VERSION": "0.1.0",` to the config dict of `build_step` in `test_surveys_golden.py`, of `config_for` in `test_surveys_step.py` and of `build_step` in `test_step_ztf_multisurvey.py`; in `test_surveys_step.py` the `assert_called_once_with` gains `"0.1.0"` after `version("feature-step")`. No other line of those files changes. `test_step_ztf.py` and `test_step_lsst.py` already set the key.

- [ ] **Step 6: Run the new file (all pass) and the package check (exit 0, 8 baseline failures, 0 new)**

- [ ] **Step 7: Commit**

`feat(feature_step): FEATURE_VERSION from the config selects the feature_version_lut row`

---

## PR body items to carry

- Behaviour that changes: a config without `FEATURE_VERSION` raises at construction; a failed version lookup raises instead of writing under `version_id` 1.
- Deployment order: the `feature_version` column must exist in the target schema before an image with this change starts.
- Not changed, not checked: `feature_step/scripts/local_sim` runners and the deployed configs. Any of them without `FEATURE_VERSION` raises at start.
