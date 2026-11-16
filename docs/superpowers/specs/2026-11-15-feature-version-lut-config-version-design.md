# feature_step: a second version, from the config, in `feature_version_lut`

Lines marked **[call]** are choices made while writing this, not things Francisco said. Edit or delete them.

## Goal

A row of `feature_version_lut` is identified today by (`version_name`, `sid`, `tid`), where `version_name` is the installed `feature-step` package version. The step config gains a version of its own (example: `0.1.0`), and the row is identified by (`version_name`, config version, `sid`, `tid`). When the step starts and that combination has no row, it inserts one with a new `version_id`. The `version_id` keeps travelling to the scribe as `features_version`, so two deployments of the same image with different config versions write features under different ids.

## Today

| piece | where | behaviour |
|---|---|---|
| table model | `libs/db-plugins-multisurvey/db_plugins/db/sql/models_pipeline.py`, `FeatureVersionLut` | `version_id`, `version_name`, `sid`, `tid`; PK (`version_id`, `sid`) |
| lookup / insert | `feature_step/features/database.py`, `get_or_create_version_id` | select by (`version_name`, `sid`, `tid`); when absent, insert with `MAX(version_id) + 1` for that (`sid`, `tid`) |
| caller | `feature_step/features/step.py`, `FeatureStep.__init__` | passes `version("feature-step")`, keeps the id as `self.extractor_version` |
| failure | `get_or_create_version_id` | any exception is logged and the function returns `1`; without a DB connection it returns `None` |

## Design

- Config key: `FEATURE_VERSION`, top level of the step config, a string. **[call]** The name is already present (and unused) in the configs of `test_step_ztf.py` and `test_step_lsst.py`.
- Column: `feature_version`, `VARCHAR`, not null, in `feature_version_lut`. **[call]** on the name.
- Rows that exist before the change take `feature_version = '1.0.0'`. The model declares that as the column's server default, so an `ALTER TABLE ... ADD COLUMN` built from the model backfills them.
- `get_or_create_version_id` takes the config version as a new argument, selects by (`version_name`, `feature_version`, `sid`, `tid`) and inserts the four values plus the new `version_id`. The `version_id` rule does not change: `MAX(version_id) + 1` for that (`sid`, `tid`).
- `FEATURE_VERSION` missing or empty: `FeatureStep.__init__` raises `ValueError` naming the key, as it does for `SURVEY`. **[call]** The alternative is to default to `1.0.0`, which would make a deployment that forgot the key reuse the old rows without anyone noticing.
- A failed lookup or insert raises instead of returning `1`. **[call]** With the fallback, a table that does not have the new column yet makes every pod write features under `version_id` 1 and log one error line. Without a DB connection the function still returns `None`.

## Approaches

| approach | trade-off | |
|---|---|---|
| New column `feature_version` in `feature_version_lut` | one `ALTER TABLE`; both versions stay queryable on their own | chosen |
| Concatenate both into `version_name` (`27.8.0+0.1.0`) | no schema change; the two versions can only be separated by parsing a string, and old rows have a different shape | rejected |
| Separate lookup table for the config version | a second id would have to travel to the scribe and into `feature`; changes the scribe and the `feature` table | rejected |

## Non-goals

1. Running the `ALTER TABLE` on any database, and deciding the final value of the old rows beyond the `1.0.0` default. Francisco: "es problema para después".
2. Any change to the scribe, to the `feature` table or to the `features_version` field of the command.
3. Any change to how `version_name` is obtained.
4. Protection against two pods inserting the same new `version_id` at the same time. It is today's behaviour; with the raise above the losing pod fails at start and finds the row on restart.
5. Changes to `charts/`, to deployed config files or to `feature_step/scripts/local_sim`.

## Acceptance criteria

1. `FeatureVersionLut` has a `feature_version` column: `VARCHAR`, not nullable, server default `'1.0.0'`. The primary key is unchanged.
2. With a row for (`version_name`, `feature_version`, `sid`, `tid`) present, `get_or_create_version_id` returns its `version_id` and inserts nothing.
3. With rows for the same `version_name`, `sid`, `tid` but another `feature_version`, it inserts one row holding the four values and `version_id = MAX(version_id) + 1` over that (`sid`, `tid`), and returns that id.
4. The same `feature_version` under a different `version_name`, `sid` or `tid` does not count as a match.
5. `FeatureStep.__init__` passes `config["FEATURE_VERSION"]` to `get_or_create_version_id` together with the package version, `sid` and `tid` of the survey.
6. `FeatureStep.__init__` raises `ValueError` naming `FEATURE_VERSION` when the key is missing or an empty string, before any DB call.
7. When the select or the insert raises, `get_or_create_version_id` raises; it does not return `1`. With `db_sql=None` it returns `None`.
8. Every new or changed line carries a `#` comment and every public function a docstring.

Criteria 2, 3, 4 and 7 are tested against a fake session that records the SQL text and parameters; no test needs docker.

## Done looks like

```
~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check feature_step
```

exits 0: no failure outside the recorded baseline of 8.
