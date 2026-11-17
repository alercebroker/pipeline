# Feature step slim: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `feature_step/features/step.py` keeps only the orchestration; scribe sending, xmatch, message preparation and config reading move to four modules, with no behaviour change.

**Architecture:** four modules of plain functions (one small class for the xmatch) next to `step.py`. The step keeps its attributes and its hooks and calls the modules. Each task moves one block and is safe to merge alone.

**Tech Stack:** Python 3.10, `unittest` + `mock`, run with `~/.pyenv/versions/feature-step/bin/python`.

**Spec:** `docs/superpowers/specs/2026-11-17-feature-step-slim-design.md`

**Branch / PR:** `refactor/feature-step-slim`, draft PR #652 against `refactor/feature-step-surveys`.

## Global Constraints

- No behaviour change. A difference found while moving code is restored, not documented: `sid == 2` skips the xmatch, a match without `catalog` is written as `allwise`, an oid missing from the batch gets `sid` 1, one flush after the last command of each list and none for an empty list.
- No committed test file changes (criterion 8). New tests go in new files under `feature_step/tests/unittest/`, seen red before the code exists.
- `FeatureStep` keeps the attributes `scribe_producer`, `xmatch_client` (only when the xmatch is on), `use_xmatch`, `feature_version`, `extractor_version`, `feature_name_lut` and the method `_get_sql_references` (criterion 6). `get_or_create_version_id`, `get_feature_name_lut` and `get_sql_references` stay imported in `step.py` and called through those names.
- Every new or changed line carries a `#` comment; every public function and class a docstring (a `def`/`class` line with a docstring below needs no `#`).
- Never commit `poetry.lock`. Nothing under `charts/`, the scribe, the schemas or `features/surveys/` changes.
- Test command, from `feature_step/`: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=<repo>/alerce_classifiers ~/.pyenv/versions/feature-step/bin/python -m pytest tests/unittest/<file> -q`.
- Green means, from the repo root: `~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check feature_step` exits 0 (baseline: 8 failures).

## Review Focus

1. A batch with matches but no object updates, or features but no object updates (LSST has no `update_object`): exactly one flush per non-empty list. Pinned in Task 1.
2. The xmatch returns nothing: no produce, no flush, and every message gets `xmatches = None`. Pinned in Task 2.
3. A match whose oid is not in the batch, or without `catalog`: `sid` 1 and `allwise`. Pinned in Task 2.
4. Two messages with the same oid: `collect_candids` concatenates their detection ids in message order; the xmatch `sid` is the one of the last message. Pinned in Tasks 2 and 3.
5. `MIN_DETECTIONS_FEATURES` given as a string (`"8"`): read as 8. `FEATURE_VERSION` unquoted in yaml: `ValueError` before any database call. Pinned in Task 4.

---

### Task 1: `scribe_sender.send_commands` (criteria 1, 2)

**Files:**
- Create: `feature_step/features/scribe_sender.py`, `feature_step/tests/unittest/test_slim_scribe_sender.py`
- Modify: `feature_step/features/step.py` (`produce_to_scribe`)

**Interfaces:**
- Produces: `send_commands(producer, topic: str, commands: list) -> None`. `producer` is the object with `.produce(topic=, value=, key=)` and `.flush()` (the step passes `self.scribe_producer.producer`).

- [ ] **Step 1: failing tests** in `test_slim_scribe_sender.py` with a `mock.Mock()` producer:
  - `test_commands_are_produced_in_order_and_flushed_once`: three commands with payload oids 3, 1, 2 -> `producer.mock_calls` equals three `call.produce(topic="t", value=json.dumps(c).encode("utf-8"), key=str(oid).encode("utf-8"))` in that order followed by one `call.flush()`.
  - `test_empty_list_does_nothing`: `producer.mock_calls == []`.
  - `test_step_sends_objects_then_features` : a `FeatureStep.__new__(FeatureStep)` with `survey.scribe_commands` mocked to return `{"update_object": [a], "upserting_features": [b, c]}` -> produce a, flush, produce b, produce c, flush.
  - `test_step_without_object_updates_flushes_once`: `{"upserting_features": [b]}` -> produce b, flush.
- [ ] **Step 2: run, expect FAIL** (`ModuleNotFoundError` for the module tests; the two step tests pass already and are kept as the behaviour pin: say so in the report).
- [ ] **Step 3: commit the test file alone**: `test(feature_step): scribe commands are sent in order with one flush per list`
- [ ] **Step 4: implement** `send_commands`; `produce_to_scribe` becomes the `scribe_commands` call plus one `send_commands` call per non-empty list (spec amendment, criterion 2). `json` is still imported in `step.py` until Task 2.
- [ ] **Step 5: run** the new file (all pass) and the package check (exit 0).
- [ ] **Step 6: commit**: `refactor(feature_step): one send_commands for the scribe produce loops`

### Task 2: `xmatch.XmatchEnricher` (criterion 3)

**Files:**
- Create: `feature_step/features/xmatch.py`, `feature_step/tests/unittest/test_slim_xmatch.py`
- Modify: `feature_step/features/step.py` (`__init__`, `pre_execute`; remove `_match_for_output`, `get_xmatch_info`, `produce_xmatch_to_scribe`), `feature_step/scripts/local_sim/equiv.py` (the line that uses `step.get_xmatch_info`)

**Interfaces:**
- Consumes: `send_commands` from Task 1.
- Produces: `class XmatchEnricher` with `__init__(self, client, catalogs: list, survey_name: str)`, `query(self, messages: list) -> list`, `attach(self, messages: list, results: list) -> None`, `commands(self, results: list, messages: list) -> list`. The step builds it as `self.xmatch = XmatchEnricher(self.xmatch_client, self.xmatch_catalogs, self.survey_name)` when the xmatch is on, and `pre_execute` does `results = self.xmatch.query(messages)`, `self.xmatch.attach(messages, results)`, `send_commands(self.scribe_producer.producer, self.scribe_topic_name, self.xmatch.commands(results, messages))`.

- [ ] **Step 1: failing tests** in `test_slim_xmatch.py` with a mocked client:
  - `test_query_asks_once_per_catalog_and_skips_sid_2`: messages with sids 0, 2, 0 and catalogs `["allwise", "gaia"]` -> two `conesearch_with_metadata` calls with `oids` as strings of the two non-2 messages, `ras`/`decs` from `meanra`/`meandec`, and the result is the concatenation.
  - `test_attach_sets_allwise_match_without_scalar_metadata`: the allwise match with `metadata={"id": "x", "w1mpro": {"a": 1}}` is attached with `metadata == {"w1mpro": {"a": 1}}`; the original match dict is not modified; a gaia match is not attached; a message without match gets `None`.
  - `test_attach_with_no_results_sets_none_everywhere`.
  - `test_commands_shape`: one command `{"step": "xmatch", "survey": <name>, "payload": {"oid": int, "sid": <sid of the message>, "catalog": ..., "dist": match["distance"], "oid_catalog": match["match_id"]}}` per match, in order.
  - `test_commands_defaults`: a match without `catalog` -> `"allwise"`; an oid absent from the messages -> `sid` 1; two messages with the same oid and different sid -> the last one.
  - `test_commands_empty_results_is_empty_list`.
- [ ] **Step 2: run, expect FAIL** (`ModuleNotFoundError`).
- [ ] **Step 3: commit the test file alone**: `test(feature_step): xmatch query, attach and scribe commands outside the step`
- [ ] **Step 4: implement** the class by moving the code; `step.py` loses `json` and the three functions; `equiv.py` calls `step.xmatch.query` where it called `step.get_xmatch_info`.
- [ ] **Step 5: run** the new file and the package check (exit 0); `grep -n "get_xmatch_info\|produce_xmatch_to_scribe\|_match_for_output" -r feature_step --include=*.py` returns nothing outside the new module's history.
- [ ] **Step 6: commit**: `refactor(feature_step): xmatch moves to features/xmatch.py`

### Task 3: `messages` helpers (criterion 5)

**Files:**
- Create: `feature_step/features/messages.py`, `feature_step/tests/unittest/test_slim_messages.py`
- Modify: `feature_step/features/step.py` (`pre_execute`, `execute`)

**Interfaces:**
- Produces: `gather_epochs(messages: list, survey) -> list`, `has_enough_detections(message: dict, minimum: int) -> bool`, `with_index_column(epochs: list, id_column: str) -> list`, `collect_candids(messages: list, id_column: str) -> dict`.

- [ ] **Step 1: failing tests** in `test_slim_messages.py`:
  - `test_gather_epochs_copies_and_sets_detections`: the returned dicts are new objects with `"detections"` from `survey.epochs_from_message(message)`; the input messages have no `"detections"` key added.
  - `test_has_enough_detections_counts_non_forced`: epochs `[{}, {"forced": False}, {"forced": True}]` -> True for minimum 2, False for 3.
  - `test_with_index_column`: `[{"measurement_id": 7, "oid": 5}]` -> `index_column == "7_5"`, other keys kept, input dict not modified.
  - `test_collect_candids_concatenates_per_oid`: two messages of oid 5 with `measurement_id` lists `[1, 2]` and `[3]`, one of oid 6 with `[4]` -> `{5: [1, 2, 3], 6: [4]}`.
- [ ] **Step 2: run, expect FAIL** (`ModuleNotFoundError`).
- [ ] **Step 3: commit the test file alone**: `test(feature_step): message preparation helpers outside the step`
- [ ] **Step 4: implement**; `pre_execute` ends with `[m for m in gather_epochs(messages, self.survey) if has_enough_detections(m, self.min_detections_features)]`; `execute` uses `collect_candids` and `with_index_column`. `candids` must hold the same lists as today for a batch with a repeated oid.
- [ ] **Step 5: run** the new file and the package check (exit 0).
- [ ] **Step 6: commit**: `refactor(feature_step): message preparation moves to features/messages.py`

### Task 4: `settings.read_settings` (criteria 4, 6, 7)

**Files:**
- Create: `feature_step/features/settings.py`, `feature_step/tests/unittest/test_slim_settings.py`
- Modify: `feature_step/features/step.py` (`__init__`), `feature_step/scripts/local_sim/*.py` only where a moved name is used

**Interfaces:**
- Produces: `@dataclass(frozen=True) class StepSettings` with `feature_version: str`, `min_detections_features: int`, `use_xmatch: bool`, `xmatch_catalogs: list`, `xmatch_config: dict`, `schema: str`, `scribe_topic: Optional[str]`; `read_settings(config: dict) -> StepSettings`.

- [ ] **Step 1: failing tests** in `test_slim_settings.py`:
  - `test_defaults`: a config with only `FEATURE_VERSION: "1.0.0"` and `SCRIBE_PRODUCER_CONFIG: {}` -> `1`, `False`, `["allwise", "gaia"]`, `{}`, `"multisurvey"`, `None`.
  - `test_values_are_read`: every field from its key (`MIN_DETECTIONS_FEATURES`, `USE_XMATCH`, `XMATCH_CATALOGS`, `XMATCH_CONFIG`, `DB_CONFIG.SCHEMA`, `SCRIBE_PRODUCER_CONFIG.TOPIC`).
  - `test_min_detections_as_string`: `"8"` -> `8`.
  - `test_feature_version_is_stripped`: `" 1.0.0 "` -> `"1.0.0"`.
  - `test_feature_version_errors`: missing, `""`, `"  "`, `1.1`, `0`, `False`, `["1.0.0"]` each raise `ValueError` whose text contains `FEATURE_VERSION`; the three messages are today's (`is missing from the step config`, `must be a quoted string, got ...`, `is empty in the step config`).
  - `test_step_py_is_only_orchestration` (criterion 7): `step.py` has at most 215 lines and contains none of `json.dumps`, `.produce(`, `conesearch`, `FEATURE_VERSION`.
- [ ] **Step 2: run, expect FAIL**.
- [ ] **Step 3: commit the test file alone**: `test(feature_step): step settings are read and validated outside the step`
- [ ] **Step 4: implement**; `__init__` calls `read_settings(self.config)` after `super().__init__` and before the survey and any database call, and keeps the attributes of criterion 6 assigned from it (`self.feature_version`, `self.use_xmatch`, `self.schema`, `self.min_detections_features`, `self.xmatch_catalogs`, `self.scribe_topic_name`). If `step.py` is over 215 lines after this, stop and report the count instead of deleting comments.
- [ ] **Step 5: run** the new file, `test_feature_version_lut.py` (13 pass, unedited) and the package check (exit 0); `python -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]" feature_step/scripts/local_sim/*.py` passes.
- [ ] **Step 6: commit**: `refactor(feature_step): step settings move to features/settings.py`

---

## PR body items to carry

- No behaviour change; the odd parts listed in the spec's non-goal 1 are kept on purpose.
- No version bump and no image: production keeps running the image of #646.
- `local_sim` scripts: only the lines that used a moved name changed; they were parsed, not run.
