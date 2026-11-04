# SN forwarder schema fields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The SN forwarder produces only the top-level keys its producer's schema declares, so the `timestamp` key apf's consumer adds no longer breaks strict Avro serialization.

**Architecture:** One method change in `StampClassifierStep.forward_sn_candidates` mirroring the existing `pre_produce` / `output_fields` pattern: read `self.sn_forward_producer.schema["fields"]` once in `__init__` into `self.sn_forward_fields` (None when the producer has no schema) and filter each alert through it before `produce`. One new test file drives the real LSST v11.1 schema through the recording producer.

**Tech Stack:** Python 3.11 venv `~/.pyenv/versions/3.11.11/envs/rubin-stamp-classifier-step/bin/python`, pytest, fastavro.

**Spec:** `docs/superpowers/specs/2026-10-29-sn-forwarder-schema-fields-design.md`

## Global Constraints

- Every new or changed line of code carries a `#` comment saying what it does; every public function and class carries a docstring (spec convention, PR #633).
- Committed test files are read-only: `tests/unit/test_sn_forwarding.py` is not edited (criterion 4).
- Work from the worktree `/Users/panchoandrades/Desktop/repos/pipeline-hunter`, package dir `rubin_stamp_classifier_step`. The venv resolves `apf`, `db_plugins` and `alerce_classifiers` from the main checkout unless `PYTHONPATH` pins the worktree: every pytest run uses
  `PYTHONPATH=<worktree>/rubin_stamp_classifier_step:<worktree>/alerce_classifiers:<worktree>/libs/apf:<worktree>/libs/db-plugins-multisurvey`.
- One commit for the task, no `poetry.lock`, no yaml with credentials.

## Review Focus

1. An alert dict missing a key the schema declares (not the case for real alerts) must still be produced as-is for that key: the filter only drops keys, never adds or defaults. Test pinned in Task 1 (`test_filter_only_drops_keys`).
2. The original `alert` dict must not be mutated (it is shared by reference with the DB and output paths). Test pinned in Task 1.
3. A forwarder producer without a `schema` attribute (recording producer without `SCHEMA_PATH`, or `DefaultProducer`) must get the alert unchanged. Test pinned in Task 1.
4. The filter must be computed once at init, not per message (128-partition topic, 10 alerts per batch). Reviewer checks by reading `__init__`.
5. The field set must come from the forwarder producer's schema, not from `self.output_fields` (the output topic schema), which would strip almost everything. Test 3 in Task 1 covers it: the output schema has no `diaSource`.

---

### Task 1: Filter forwarded alerts by the forwarder schema

**Files:**
- Modify: `rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py` (`__init__` lines 82-92, `forward_sn_candidates` lines 286-302)
- Create: `rubin_stamp_classifier_step/tests/unit/test_sn_forwarder_schema.py`

**Interfaces:**
- Consumes: `StampClassifierStep.forward_sn_candidates(predictions: List[dict])`, `self.sn_forward_producer` (apf producer; `KafkaProducer` and `tests.unit.recording_producer.RecordingProducer` expose `.schema` when configured with `SCHEMA_PATH`), `make_step(forward_config=...)` from `tests/unit/test_sn_forwarding.py`.
- Produces: `self.sn_forward_fields: set[str] | None` on the step.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_sn_forwarder_schema.py`:

```python
"""The SN forwarder emits only the keys the forwarder's schema declares (spec 2026-10-29, criteria 1-3)."""
import io
import os

import fastavro
import fastavro.schema
import pytest

from tests.unit.test_sn_forwarding import make_step, FORWARD_TOPIC

LSST_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "schemas", "surveys", "lsst_v11.1", "lsst.v11_1.alert.avsc"
)


def minimal_record(schema, named=None):
    """Smallest value fastavro accepts for a parsed schema: declared defaults, zero values elsewhere."""
    named = named if named is not None else {}
    if isinstance(schema, list):
        return None if "null" in schema else minimal_record(schema[0], named)
    if isinstance(schema, str):
        if schema in named:
            return minimal_record(named[schema], named)
        return {"null": None, "boolean": False, "int": 0, "long": 0, "float": 0.0,
                "double": 0.0, "bytes": b"", "string": ""}[schema]
    kind = schema["type"]
    if kind == "record":
        named[schema["name"]] = schema
        return {f["name"]: (f["default"] if "default" in f else minimal_record(f["type"], named))
                for f in schema["fields"]}
    if kind == "array":
        return []
    if kind == "map":
        return {}
    if kind == "enum":
        return schema["symbols"][0]
    if kind == "fixed":
        return b"\x00" * schema["size"]
    return minimal_record(kind, named)


@pytest.fixture
def lsst_schema():
    return fastavro.schema.load_schema(LSST_SCHEMA_PATH)


@pytest.fixture
def schema_step():
    return make_step(forward_config={
        "CLASS": "tests.unit.recording_producer.RecordingProducer",
        "TOPIC": FORWARD_TOPIC,
        "SCHEMA_PATH": LSST_SCHEMA_PATH,
    })


def sn_prediction(alert):
    return {"oid": 7, "sid": 1, "probabilities": {"SN": 0.9, "AGN": 0.1}, "alert": alert}


def test_forwarded_alert_with_consumer_timestamp_serializes_strictly(schema_step, lsst_schema):
    alert = dict(minimal_record(lsst_schema), timestamp=1761750000000)  # what apf's consumer hands the step
    schema_step.forward_sn_candidates([sn_prediction(alert)])
    (produced, kwargs), = schema_step.sn_forward_producer.produced
    fastavro.schemaless_writer(io.BytesIO(), lsst_schema, produced, strict=True)  # raises before the fix
    assert "timestamp" not in produced
    assert kwargs["key"] == "7"


def test_filter_does_not_mutate_the_alert(schema_step, lsst_schema):
    alert = dict(minimal_record(lsst_schema), timestamp=1)
    schema_step.forward_sn_candidates([sn_prediction(alert)])
    assert "timestamp" in alert


def test_filter_only_drops_keys(schema_step, lsst_schema):
    alert = {"diaSourceId": 5, "timestamp": 1}  # schema keys missing on purpose
    schema_step.forward_sn_candidates([sn_prediction(alert)])
    (produced, _), = schema_step.sn_forward_producer.produced
    assert produced == {"diaSourceId": 5}


def test_fields_come_from_the_forwarder_schema_not_the_output_schema(schema_step):
    assert "diaSource" in schema_step.sn_forward_fields
    assert schema_step.sn_forward_fields != schema_step.output_fields


def test_producer_without_schema_gets_the_alert_unchanged():
    step = make_step(forward_config={"CLASS": "tests.unit.recording_producer.RecordingProducer", "TOPIC": FORWARD_TOPIC})
    alert = {"diaSourceId": 5, "timestamp": 1}
    step.forward_sn_candidates([sn_prediction(alert)])
    (produced, _), = step.sn_forward_producer.produced
    assert produced == alert
    assert step.sn_forward_fields is None
```

- [ ] **Step 2: Run them to verify they fail**

Run from `<worktree>/rubin_stamp_classifier_step` with the PYTHONPATH above:
`<venv>/bin/python -m pytest tests/unit/test_sn_forwarder_schema.py -q`
Expected: the first test fails with `ValueError: record contains more fields than the schema specifies: timestamp`; the second passes (nothing mutates today); the third fails (`timestamp` still present); the fourth and fifth fail with `AttributeError: ... sn_forward_fields`.

- [ ] **Step 3: Implement**

In `__init__`, right after `self.sn_forward_producer = get_class(...)(forward_cfg)`: load the schema the same way `output_fields` is built from `self.producer` (lines 54-57) into `self.sn_forward_fields` (set of top-level field names, `None` when the producer has no `schema`). Initialize `self.sn_forward_fields = None` next to `self.sn_forward_producer = None` so the attribute always exists.

In `forward_sn_candidates`, replace `self.sn_forward_producer.produce(msg["alert"], key=str(msg["oid"]))` with a produce of a new dict holding only the keys in `self.sn_forward_fields` when that is not `None`, else `msg["alert"]` itself. Build the new dict with a comprehension (no `pop`, no mutation). Update the docstring: the alert is re-emitted with the consumer's extra keys stripped. A `#` comment on every new or changed line.

- [ ] **Step 4: Run the new tests, then the package delta**

`<venv>/bin/python -m pytest tests/unit/test_sn_forwarder_schema.py tests/unit/test_sn_forwarding.py -q` → all pass.
From the worktree root: `PYTHONPATH=<as above> ~/.pyenv/versions/feature-step/bin/python .claude/scripts/test_delta.py check rubin_stamp_classifier_step` → exit 0 (the script lives in the main checkout at `/Users/panchoandrades/Desktop/repos/pipeline/.claude/scripts/test_delta.py`; run it with the worktree as cwd so it resolves the worktree root and baseline).

- [ ] **Step 5: Commit**

```bash
git add rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py rubin_stamp_classifier_step/tests/unit/test_sn_forwarder_schema.py
git commit -m "fix(rubin_stamp_classifier_step): SN forwarder emits only the forwarder schema's fields"
```
