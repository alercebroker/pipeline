# feature_step ZTF Multisurvey Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `feature_step` correctly consume a real `magstats_ms_ztf` message, so the ZTF multisurvey path produces the same 127 features the deployed `feature_name_lut` was seeded with.

**Architecture:** `pre_execute` becomes the single message-shaping point — it merges `detections + previous_detections + forced_photometries` into one bogus-filtered list, exactly like the LSST arm merges `sources + previous_sources`. `execute` then hands that one list to `detections_to_astro_object` with `forced=[]`, and the parser collapses to a single loop where the per-row `forced` flag does the routing. Everything downstream (reference/bogus aux frames, corrected-magnitude coalesce, W1–W4, feature ids) follows from that one list.

**Tech Stack:** Python 3.10, pandas 2.x, `lc_classifier` (local path dep), `apf_base`, pytest.

**Source spec:** `docs/superpowers/specs/2026-08-04-feature-step-ztf-multisurvey-design.md`. Read it before starting — it carries the rationale and the DB-state verification behind every change here.

---

## Environment: how to run the tests

There is no poetry env on this machine. Use the `feature_step` conda env with the
repo's packages forced ahead of the ones its `.pth` files point at (they point at
a *different* checkout, `desktop/online/pipeline`, which would silently be used
otherwise).

Define this once per shell:

```bash
export REPO=/home/fandrades/desktop/pipeline_features/pipeline
export PY=/home/fandrades/miniconda3/envs/feature_step/bin/python
export PYTHONPATH=$REPO/lc_classifier:$REPO/libs/apf:$REPO/libs/xmatch_client:$REPO/libs/db-plugins-multisurvey:$REPO/libs/test_utils
```

Run feature_step tests from `$REPO/feature_step`:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py -q -p no:warnings
```

Run lc_classifier tests from `$REPO/lc_classifier`:

```bash
cd $REPO/lc_classifier && $PY -m pytest tests/features/test_discard_bogus_detections.py -q -p no:warnings
```

**Sanity check before Task 1** — confirm the import override works:

```bash
$PY -c "import lc_classifier, apf; print(lc_classifier.__file__)"
```

Expected: a path under `/home/fandrades/desktop/pipeline_features/pipeline/lc_classifier/`.
If it prints a `desktop/online/pipeline` path, `PYTHONPATH` is wrong — fix it before proceeding.

### Baseline (measured 2026-08-06, before any change)

`tests/unittest` is **already partly red** and that is not your doing:

```
tests/unittest/test_step_lsst.py   1 failed, 1 passed
tests/unittest/test_step_ztf.py    7 failed, 3 passed
```

`tests/unittest/test_step_ztf.py` and `tests/message_factory.py` build the legacy
`extra_fields` shape and are **left untouched by this plan** (the user's explicit
call). Your job is to not make that number worse: after every task, the legacy
suite must still be exactly `8 failed, 4 passed`. A verification step at the end
of each code task checks this.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `lc_classifier/lc_classifier/features/core/base.py` | `discard_bogus_detections` — tolerate int `procstatus` | Modify (1 hunk) |
| `lc_classifier/lc_classifier/features/extractors/spm_extractor.py` | emit `SPM_mjd_ref`; version → `1.0.2` | Replace with sibling-repo copy |
| `lc_classifier/lc_classifier/features/extractors/tde_extractor.py` | emit `TDE_mjd_ref`, `fleet_mjd_ref`; versions → `1.0.2` / `1.0.3` | Replace with sibling-repo copy |
| `lc_classifier/lc_classifier/features/extractors/ulens_extractor.py` | emit `ulens_mjd_ref`; version → `1.0.3` | Replace with sibling-repo copy |
| `lc_classifier/tests/features/test_discard_bogus_detections.py` | procstatus/rb filter unit test | Create |
| `feature_step/features/step.py` | `pre_execute` flatten; `execute` stamps `aid`, passes `forced=[]` | Modify (2 hunks) |
| `feature_step/features/utils/parsers.py` | single-loop parser, corrected-mag coalesce, xmatch gate, LUT-driven feature ids, oid precision | Modify (6 hunks) |
| `feature_step/tests/message_factory_ztf_ms.py` | build `magstats_ms_ztf` messages from the Avro schema | Create |
| `feature_step/tests/unittest/test_step_ztf_multisurvey.py` | the ZTF multisurvey test suite | Create |

Task order is: fixture → lc_classifier port → parser → step. The fixture comes
first because every later test needs it. The lc_classifier port comes next
because it is a pure file copy with no dependency on the parser work.

---

## Task 1: ZTF multisurvey message fixture

Nothing else can be tested without a message that matches
`schemas/magstats_ms_step/ztf/output.avsc`. Three arrays, three field
vocabularies, flat records, **no `extra_fields` anywhere**.

**Files:**
- Create: `feature_step/tests/message_factory_ztf_ms.py`

- [ ] **Step 1: Write the fixture factory**

Create `feature_step/tests/message_factory_ztf_ms.py`:

```python
"""Build `magstats_ms_ztf` messages (schemas/magstats_ms_step/ztf/output.avsc).

Flat records, no `extra_fields`. Three arrays with distinct vocabularies:
`detections` (candidate), `previous_detections` (prv_candidate) and
`forced_photometries` (forced_photometry).
"""
import random

ZTF_SID = 0
ZTF_TID = 0
BAND_MAP = {"g": 1, "r": 2, "i": 3}


def _base_epoch(oid, measurement_id, band, mjd, forced, rng):
    return {
        "new": True,
        "oid": oid,
        "sid": ZTF_SID,
        "tid": ZTF_TID,
        "pid": rng.randint(1, 999999),
        "band": BAND_MAP[band],
        "measurement_id": measurement_id,
        "mjd": mjd,
        "ra": 250.0 + rng.random() * 1e-4,
        "e_ra": 0.0001,
        "dec": 30.0 + rng.random() * 1e-4,
        "e_dec": 0.0001,
        "mag": 18.0 + rng.random(),
        "e_mag": 0.05 + rng.random() * 0.05,
        "isdiffpos": 1,
        "forced": forced,
        "parent_candid": None,
        "corrected": True,
        "dubious": False,
        "stellar": False,
        "diffmaglim": 20.5,
    }


def candidate(oid, measurement_id, band, mjd, rng, rb=0.9, rfid=783120150):
    """One `candidate` record: rb, PS1 columns, magpsf_corr/sigmapsf_corr_ext."""
    epoch = _base_epoch(oid, measurement_id, band, mjd, False, rng)
    epoch.update(
        {
            "has_stamp": True,
            "magpsf_corr": 17.5 + rng.random(),
            "sigmapsf_corr": 0.05,
            "sigmapsf_corr_ext": 0.06 + rng.random() * 0.02,
            "rb": rb,
            "distnr": 0.3 + rng.random() * 0.2,
            "magnr": 17.0,
            "sigmagnr": 0.02,
            "chinr": 0.5,
            "sharpnr": -0.02,
            "rfid": rfid,
            "sgscore1": 0.1,
            "sgmag1": 18.1,
            "srmag1": 17.9,
            "simag1": 17.8,
            "szmag1": 17.7,
            "distpsnr1": 0.4,
            "rbversion": "t17_f5_c3",
            "drbversion": "d6_m7",
        }
    )
    return epoch


def prv_candidate(oid, measurement_id, band, mjd, rng, rb=0.9):
    """One `prv_candidate` record: rb but no rfid and no PS1 columns."""
    epoch = _base_epoch(oid, measurement_id, band, mjd, False, rng)
    epoch.update(
        {
            "has_stamp": False,
            "magpsf_corr": 17.5 + rng.random(),
            "sigmapsf_corr": 0.05,
            "sigmapsf_corr_ext": 0.06 + rng.random() * 0.02,
            "rb": rb,
            "distnr": 0.3 + rng.random() * 0.2,
            "magnr": 17.0,
            "sigmagnr": 0.02,
            "chinr": 0.5,
            "sharpnr": -0.02,
            "rbversion": "t17_f5_c3",
        }
    )
    return epoch


def forced_photometry(oid, measurement_id, band, mjd, rng, procstatus="0",
                      rfid=783120150):
    """One `forced_photometry` record: procstatus, mag_corr/e_mag_corr_ext, no rb."""
    epoch = _base_epoch(oid, measurement_id, band, mjd, True, rng)
    epoch.update(
        {
            "mag_corr": 17.5 + rng.random(),
            "e_mag_corr": 0.05,
            "e_mag_corr_ext": 0.06 + rng.random() * 0.02,
            "procstatus": procstatus,
            "distnr": 0.3 + rng.random() * 0.2,
            "magnr": 17.0,
            "sigmagnr": 0.02,
            "chinr": 0.5,
            "sharpnr": -0.02,
            "rfid": rfid,
            "ranr": 250.0,
            "decnr": 30.0,
            "programid": 1,
            "forcediffimflux": 100.0,
            "forcediffimfluxunc": 10.0,
        }
    )
    return epoch


def non_detection(oid, band, mjd):
    return {
        "oid": oid,
        "sid": ZTF_SID,
        "tid": ZTF_TID,
        "band": BAND_MAP[band],
        "mjd": mjd,
        "diffmaglim": 20.5,
    }


def generate_message(
    oid=36028941624528297,
    bands=("g", "r"),
    n_detections=6,
    n_previous_detections=4,
    n_forced=5,
    seed=42,
    with_xmatch=False,
):
    rng = random.Random(seed)
    mjd = 60000.0
    mid = 1000

    detections, previous_detections, forced = [], [], []
    for i in range(n_detections):
        band = bands[i % len(bands)]
        detections.append(candidate(oid, mid, band, mjd, rng))
        mjd += 1.7
        mid += 1
    for i in range(n_previous_detections):
        band = bands[i % len(bands)]
        previous_detections.append(prv_candidate(oid, mid, band, mjd, rng))
        mjd += 1.7
        mid += 1
    for i in range(n_forced):
        band = bands[i % len(bands)]
        forced.append(forced_photometry(oid, mid, band, mjd, rng))
        mjd += 1.7
        mid += 1

    message = {
        "oid": oid,
        "sid": ZTF_SID,
        "measurement_id": [d["measurement_id"] for d in detections],
        "meanra": 250.0,
        "meandec": 30.0,
        "detections": detections,
        "previous_detections": previous_detections,
        "forced_photometries": forced,
        "non_detections": [non_detection(oid, "g", 59990.0)],
    }
    if with_xmatch:
        message["xmatches"] = allwise_match(oid)
    return message


def allwise_match(oid, w1=15.1, w2=14.9, w3=12.5, w4=9.1):
    """The shape `XmatchClient.conesearch_with_metadata` returns."""
    return {
        "oid": str(oid),
        "catalog": "allwise",
        "distance": 0.5,
        "match_id": "J000000.00+000000.0",
        "metadata": {
            "w1mpro": {"Float64": w1},
            "w2mpro": {"Float64": w2},
            "w3mpro": {"Float64": w3},
            "w4mpro": {"Float64": w4},
        },
    }
```

No batch helper: the two tasks that need several messages build the list inline,
and a `generate_input_batch(n, **kwargs)` wrapper would collide with the `oid` and
`seed` it has to set per item.

Notes for reviewers: `oid=36028941624528297` is a real ZTF multisurvey oid and is
deliberately larger than `2**53` — Task 8 depends on that.

- [ ] **Step 2: Verify the fixture matches the Avro schema**

Run:

```bash
cd $REPO/feature_step && $PY - <<'EOF'
import json, random
from tests.message_factory_ztf_ms import generate_message

msg = generate_message()
base = "../schemas/magstats_ms_step/ztf"
records = {
    "detections": "candidate.avsc",
    "previous_detections": "prv_candidate.avsc",
    "forced_photometries": "forced_photometry.avsc",
    "non_detections": "non_detection.avsc",
}
for key, filename in records.items():
    schema = json.load(open(f"{base}/{filename}"))
    allowed = {f["name"] for f in schema["fields"]}
    for row in msg[key]:
        extra = set(row) - allowed
        assert not extra, f"{key}: fields not in {filename}: {sorted(extra)}"
    print(key, "ok", len(msg[key]), "rows")

output = json.load(open(f"{base}/output.avsc"))
top = {f["name"] for f in output["fields"]}
assert set(msg) <= top, sorted(set(msg) - top)
print("top-level ok")
EOF
```

Expected output:

```
detections ok 6 rows
previous_detections ok 4 rows
forced_photometries ok 5 rows
non_detections ok 1 rows
top-level ok
```

(Every field the factory emits exists in the schema. The factory does not fill
every optional column — that is intentional: real messages carry nulls too.)

- [ ] **Step 3: Commit**

```bash
cd $REPO && git add feature_step/tests/message_factory_ztf_ms.py
git commit -m "test(feature_step): magstats_ms_ztf message factory"
```

---

## Task 2: `discard_bogus_detections` tolerates int `procstatus`

`procstatus` arrives as `["null", "string"]` in the Avro schema but as an int
from the DB and from some producers. The filter compares against the string
literals `"0"` / `"57"`, so an int `0` is currently treated as bogus and the
epoch is silently dropped.

**Files:**
- Create: `lc_classifier/tests/features/test_discard_bogus_detections.py`
- Modify: `lc_classifier/lc_classifier/features/core/base.py:121-126`

- [ ] **Step 1: Write the failing test**

Create `lc_classifier/tests/features/test_discard_bogus_detections.py`:

```python
import unittest

from lc_classifier.features.core.base import discard_bogus_detections


def forced(measurement_id, procstatus):
    return {"measurement_id": measurement_id, "forced": True, "procstatus": procstatus}


def detection(measurement_id, rb):
    return {"measurement_id": measurement_id, "forced": False, "rb": rb}


class TestDiscardBogusDetections(unittest.TestCase):
    def test_str_procstatus(self):
        epochs = [forced(1, "0"), forced(2, "57"), forced(3, "2")]
        kept = [d["measurement_id"] for d in discard_bogus_detections(epochs)]
        self.assertEqual([1, 2], kept)

    def test_int_procstatus_is_coerced(self):
        epochs = [forced(1, 0), forced(2, 57), forced(3, 2)]
        kept = [d["measurement_id"] for d in discard_bogus_detections(epochs)]
        self.assertEqual([1, 2], kept)

    def test_missing_procstatus_is_kept(self):
        kept = discard_bogus_detections([{"measurement_id": 1, "forced": True}])
        self.assertEqual(1, len(kept))

    def test_low_rb_detection_is_dropped(self):
        epochs = [detection(1, 0.9), detection(2, 0.1)]
        kept = [d["measurement_id"] for d in discard_bogus_detections(epochs)]
        self.assertEqual([1], kept)

    def test_rb_is_only_applied_to_non_forced_rows(self):
        epochs = [{"measurement_id": 1, "forced": True, "rb": 0.1, "procstatus": "0"}]
        self.assertEqual(1, len(discard_bogus_detections(epochs)))

    def test_extra_fields_shape_still_supported(self):
        epochs = [
            {"measurement_id": 1, "forced": False, "extra_fields": {"rb": 0.9}},
            {"measurement_id": 2, "forced": False, "extra_fields": {"rb": 0.1}},
        ]
        kept = [d["measurement_id"] for d in discard_bogus_detections(epochs)]
        self.assertEqual([1], kept)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd $REPO/lc_classifier && $PY -m pytest tests/features/test_discard_bogus_detections.py -q -p no:warnings
```

Expected: `1 failed, 5 passed` — `test_int_procstatus_is_coerced` fails with
`AssertionError: [] != [1, 2]`.

- [ ] **Step 3: Apply the fix**

In `lc_classifier/lc_classifier/features/core/base.py`, inside
`discard_bogus_detections`, replace:

```python
        mask_procstatus = (
            procstatus is not None
            and det["forced"]
            and (procstatus != "0")
            and (procstatus != "57")
        )
```

with:

```python
        # procstatus may arrive as int or str; coerce before the string compare so
        # valid forced epochs (procstatus 0 or 57) are not discarded.
        mask_procstatus = (
            procstatus is not None
            and det["forced"]
            and (str(procstatus) != "0")
            and (str(procstatus) != "57")
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
cd $REPO/lc_classifier && $PY -m pytest tests/features/test_discard_bogus_detections.py -q -p no:warnings
```

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
cd $REPO && git add lc_classifier/lc_classifier/features/core/base.py lc_classifier/tests/features/test_discard_bogus_detections.py
git commit -m "fix(lc_classifier): tolerate int procstatus in discard_bogus_detections"
```

---

## Task 3: Port the `*_mjd_ref` extractor features

The deployed `multisurvey_ztf.feature_name_lut` (sid=0, tid=0) has **127 rows**
and was seeded from the extractor set in `desktop/pipeline` commit `8743448fa`.
Today's extractors emit 123 names — the four `*_mjd_ref` ids would never be
written. The three local files differ from the sibling repo's **only** by that
commit (verified with `diff`), so this is a straight file copy.

**Files:**
- Modify: `lc_classifier/lc_classifier/features/extractors/spm_extractor.py`
- Modify: `lc_classifier/lc_classifier/features/extractors/tde_extractor.py`
- Modify: `lc_classifier/lc_classifier/features/extractors/ulens_extractor.py`

- [ ] **Step 1: Confirm the files differ only by the intended commit**

Run:

```bash
for f in spm_extractor tde_extractor ulens_extractor; do
  echo "=== $f"
  diff $REPO/lc_classifier/lc_classifier/features/extractors/$f.py \
       /home/fandrades/desktop/pipeline/lc_classifier/lc_classifier/features/extractors/$f.py
done
```

Expected: only the hunks below. **If any other hunk appears, stop and report it**
— the copy in Step 2 would then carry unrelated changes.

- `spm_extractor.py`: `self.version` `"1.0.1"` → `"1.0.2"`, plus a `SPM_mjd_ref` block after the `SPM_chi` loop.
- `tde_extractor.py`: `TDETailExtractor.version` `"1.0.1"` → `"1.0.2"` with `TDE_mjd_ref` in both branches; `FleetExtractor.version` `"1.0.2"` → `"1.0.3"` with `fleet_mjd_ref` in all three branches.
- `ulens_extractor.py`: `MicroLensExtractor.version` `"1.0.2"` → `"1.0.3"`; `get_observations` returns `(observations, mjd_first_detection)`; `any_band_fit` guard; `ulens_mjd_ref` appended.

- [ ] **Step 2: Copy the three files**

```bash
for f in spm_extractor tde_extractor ulens_extractor; do
  cp /home/fandrades/desktop/pipeline/lc_classifier/lc_classifier/features/extractors/$f.py \
     $REPO/lc_classifier/lc_classifier/features/extractors/$f.py
done
```

- [ ] **Step 3: Verify the four names are emitted**

Run:

```bash
cd $REPO/lc_classifier && $PY - <<'EOF'
import inspect
from lc_classifier.features.extractors.spm_extractor import SPMExtractor
from lc_classifier.features.extractors.tde_extractor import TDETailExtractor, FleetExtractor
from lc_classifier.features.extractors.ulens_extractor import MicroLensExtractor

print("SPM", SPMExtractor(bands=list("gr"), unit="diff_flux", redshift=None,
                          extinction_color_excess=None, forced_phot_prelude=30.0).version)
print("TDETail", TDETailExtractor.version)
print("Fleet", FleetExtractor.version)
print("MicroLens", MicroLensExtractor.version)
for cls, name in [(SPMExtractor, "SPM_mjd_ref"), (TDETailExtractor, "TDE_mjd_ref"),
                  (FleetExtractor, "fleet_mjd_ref"), (MicroLensExtractor, "ulens_mjd_ref")]:
    assert name in inspect.getsource(cls), name
    print(name, "present")
EOF
```

Expected:

```
SPM 1.0.2
TDETail 1.0.2
Fleet 1.0.3
MicroLens 1.0.3
SPM_mjd_ref present
TDE_mjd_ref present
fleet_mjd_ref present
ulens_mjd_ref present
```

(The end-to-end assertion that this yields exactly 127 unique names lands in
Task 9. This step only confirms the copy landed.)

- [ ] **Step 4: Commit**

```bash
cd $REPO && git add lc_classifier/lc_classifier/features/extractors/
git commit -m "feat(lc_classifier): emit reference-epoch features (*_mjd_ref) from extractors

Ported from desktop/pipeline 8743448fa. The deployed multisurvey_ztf
feature_name_lut was seeded from that extractor set; without this the ZTF
step emits 123 of the 127 seeded features."
```

---

## Task 4: `pre_execute` merges the three ZTF arrays

This is the change every later ZTF test depends on. It mirrors the LSST arm
(`step.py:325`, which merges `sources + previous_sources`) and the offline
harness (`features/offline/lc_features.py::_prepare_detections`).

Consequences, all covered by the tests below: `previous_detections` reaches the
extractor for the first time; `discard_bogus_detections` finally sees forced
rows, so the `procstatus` filter actually runs; a null `forced_photometries` no
longer blows up.

**Files:**
- Modify: `feature_step/features/step.py:319-323`
- Create: `feature_step/tests/unittest/test_step_ztf_multisurvey.py`

- [ ] **Step 1: Write the failing tests**

Create `feature_step/tests/unittest/test_step_ztf_multisurvey.py` with the module
header, helpers and `PreExecuteTestCase` (later tasks append the other classes to
this same file):

```python
"""ZTF multisurvey path: `magstats_ms_ztf` message -> features.

Covers the flat, three-array input contract (no `extra_fields`) that
`schemas/magstats_ms_step/ztf/output.avsc` defines.
"""
import logging
import random
import unittest
from unittest import mock

import pandas as pd

from features.step import FeatureStep
from features.utils.parsers import (
    detections_to_astro_object,
    prepare_ao_features_for_db,
)
from lc_classifier.features.core.base import query_ao_table
from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor

from ..message_factory_ztf_ms import (
    allwise_match,
    candidate,
    forced_photometry,
    generate_message,
)

CONSUMER_CONFIG = {
    "CLASS": "unittest.mock.MagicMock",
    "PARAMS": {"bootstrap.servers": "server", "group.id": "group_id"},
    "TOPICS": ["topic"],
}
PRODUCER_CONFIG = {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test"}
SCRIBE_PRODUCER_CONFIG = {"CLASS": "unittest.mock.MagicMock", "TOPIC": "test-scribe"}


def build_step(**extra_config):
    config = {
        "PRODUCER_CONFIG": PRODUCER_CONFIG,
        "CONSUMER_CONFIG": CONSUMER_CONFIG,
        "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
        "SURVEY": "ztf",
        **extra_config,
    }
    step = FeatureStep(config=config, db_sql=mock.MagicMock())
    # `produce_to_scribe` reaches through to `scribe_producer.producer.produce`,
    # so a plain MagicMock (not an autospec of GenericProducer) is what fits.
    step.scribe_producer = mock.MagicMock()
    # The mocked db_sql makes these two MagicMocks; pin them so scribe commands
    # stay JSON-serializable.
    step.extractor_version = 1
    step.feature_name_lut = {}
    return step


def astro_object_from(message, xmatches=None, references_db=None):
    """pre_execute + execute packing, without Kafka."""
    step = build_step()
    prepared = step.pre_execute([message])[0]
    epochs = [
        {**e, "aid": e["oid"], "index_column": f'{e["measurement_id"]}_{e["oid"]}'}
        for e in prepared["detections"]
    ]
    return detections_to_astro_object(epochs, [], xmatches, references_db)


class PreExecuteTestCase(unittest.TestCase):
    def test_three_arrays_are_merged_into_detections(self):
        message = generate_message(
            n_detections=6, n_previous_detections=4, n_forced=5
        )
        result = build_step().pre_execute([message])[0]

        self.assertEqual(15, len(result["detections"]))
        measurement_ids = {d["measurement_id"] for d in result["detections"]}
        for source in ("detections", "previous_detections", "forced_photometries"):
            for epoch in message[source]:
                self.assertIn(epoch["measurement_id"], measurement_ids)

    def test_missing_forced_photometries_is_not_none(self):
        message = generate_message()
        message["forced_photometries"] = None
        result = build_step().pre_execute([message])[0]

        self.assertEqual(10, len(result["detections"]))

        del message["forced_photometries"]
        result = build_step().pre_execute([message])[0]
        self.assertEqual(10, len(result["detections"]))

    def test_forced_rows_outside_allowed_procstatus_are_dropped(self):
        rng = random.Random(1)
        oid = 36028941624528297
        message = generate_message(oid=oid)
        message["detections"] = [candidate(oid, 1, "g", 60000.0, rng, rb=0.9)]
        message["previous_detections"] = []
        message["forced_photometries"] = [
            forced_photometry(oid, 10, "g", 60001.0, rng, procstatus="0"),
            forced_photometry(oid, 11, "g", 60002.0, rng, procstatus="57"),
            forced_photometry(oid, 12, "g", 60003.0, rng, procstatus="2"),
            forced_photometry(oid, 13, "g", 60004.0, rng, procstatus=0),
            forced_photometry(oid, 14, "g", 60005.0, rng, procstatus=2),
        ]

        result = build_step().pre_execute([message])[0]

        kept = [d["measurement_id"] for d in result["detections"]]
        self.assertEqual([1, 10, 11, 13], sorted(kept))

    def test_low_rb_detections_are_dropped(self):
        rng = random.Random(2)
        oid = 36028941624528297
        message = generate_message(oid=oid)
        message["detections"] = [
            candidate(oid, 1, "g", 60000.0, rng, rb=0.9),
            candidate(oid, 2, "g", 60001.0, rng, rb=0.1),
        ]
        message["previous_detections"] = []
        message["forced_photometries"] = []

        result = build_step().pre_execute([message])[0]

        self.assertEqual([1], [d["measurement_id"] for d in result["detections"]])

    def test_min_detections_counts_only_non_forced_rows(self):
        message = generate_message(
            n_detections=1, n_previous_detections=0, n_forced=8
        )
        step = build_step(MIN_DETECTIONS_FEATURES=2)

        self.assertEqual(0, len(step.pre_execute([message])))

        message = generate_message(
            n_detections=2, n_previous_detections=0, n_forced=0
        )
        self.assertEqual(1, len(step.pre_execute([message])))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py -q -p no:warnings
```

Expected: `3 failed, 2 passed`. The three failures are
`test_three_arrays_are_merged_into_detections` (`15 != 6`),
`test_missing_forced_photometries_is_not_none` (`10 != 6`) and
`test_forced_rows_outside_allowed_procstatus_are_dropped` (`[1] != [1, 10, 11, 13]`).

- [ ] **Step 3: Implement the flatten**

In `feature_step/features/step.py`, inside `pre_execute`, replace:

```python
            if self.survey == "ztf":
                filtered_message["detections"] = discard_bogus_detections(
                    filtered_message.get("detections", [])
                )
                filtered_messages.append(filtered_message)
```

with:

```python
            if self.survey == "ztf":
                epochs = (
                    (message.get("detections") or [])
                    + (message.get("previous_detections") or [])
                    + (message.get("forced_photometries") or [])
                )
                filtered_message["detections"] = discard_bogus_detections(epochs)
                filtered_messages.append(filtered_message)
```

All three reads use `or []` rather than a `.get` default. The schema declares
each array as required (`{"type": "array", ...}`, not a `["null", ...]` union),
so neither form should ever matter — but `or []` also absorbs an explicit `None`,
and applying it uniformly avoids an asymmetry that would read as if two of the
three lines had been left deliberately unguarded.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py -q -p no:warnings
```

Expected: `5 passed`.

- [ ] **Step 5: Confirm the legacy suite is unchanged**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf.py tests/unittest/test_step_lsst.py -q -p no:warnings
```

Expected: `8 failed, 4 passed` — identical to the baseline at the top of this plan.

- [ ] **Step 6: Commit**

```bash
cd $REPO && git add feature_step/features/step.py feature_step/tests/unittest/test_step_ztf_multisurvey.py
git commit -m "feat(feature_step): merge the three ZTF arrays in pre_execute

previous_detections now reaches the extractor and the procstatus bogus
filter finally sees forced rows."
```

---

## Task 5: Single-loop parser with the corrected-magnitude coalesce

Two changes that must land together, because each breaks the other if split.

**The alignment bug (#4/#5).** `get_reference_for_each_detection` and
`get_bogus_flags_for_each_detection` are computed over `detections` only, but the
frame `a` is built from `detections + forced`. The `pd.concat(..., axis=1)` at
`parsers.py:277` and `:282` then aligns two frames of different lengths, so every
forced row silently gets `NaN` for `distnr`, `rfid`, `rb` and `procstatus`. One
loop over one list makes that structurally impossible.

**The corrected-magnitude coalesce (#1).** `candidate`/`prv_candidate` carry
`magpsf_corr`/`sigmapsf_corr_ext`; `forced_photometry` carries
`mag_corr`/`e_mag_corr_ext`. Each row populates exactly one pair. Both spellings
must be selected into the frame and merged **before** the `DETECTION_KEYS_MAP`
rename — which means the two `mag_corr` entries must come out of that map, or the
rename would create duplicate column names.

**Files:**
- Modify: `feature_step/features/utils/parsers.py:11-28` (`DETECTION_KEYS_MAP`)
- Modify: `feature_step/features/utils/parsers.py:233-271` (`detections_to_astro_object`)
- Modify: `feature_step/tests/unittest/test_step_ztf_multisurvey.py`

- [ ] **Step 1: Write the failing tests**

Append to `feature_step/tests/unittest/test_step_ztf_multisurvey.py`, **before**
the `if __name__ == "__main__":` block:

```python
class ParserTestCase(unittest.TestCase):
    def test_forced_epochs_keep_their_corrected_magnitude(self):
        message = generate_message(n_detections=4, n_previous_detections=2, n_forced=3)
        ao = astro_object_from(message)

        forced_mag = ao.forced_photometry[ao.forced_photometry["unit"] == "magnitude"]
        self.assertEqual(3, len(forced_mag))
        self.assertTrue(forced_mag["brightness"].notna().all())
        self.assertTrue(forced_mag["e_brightness"].notna().all())

        expected = sorted(f["mag_corr"] for f in message["forced_photometries"])
        self.assertEqual(expected, sorted(forced_mag["brightness"].tolist()))

    def test_detections_keep_their_corrected_magnitude(self):
        message = generate_message(n_detections=4, n_previous_detections=2, n_forced=3)
        ao = astro_object_from(message)

        det_mag = ao.detections[ao.detections["unit"] == "magnitude"]
        self.assertEqual(6, len(det_mag))
        self.assertTrue(det_mag["brightness"].notna().all())

        expected = sorted(
            d["magpsf_corr"]
            for d in message["detections"] + message["previous_detections"]
        )
        self.assertEqual(expected, sorted(det_mag["brightness"].tolist()))

    def test_previous_detections_reach_the_astro_object(self):
        message = generate_message(n_detections=3, n_previous_detections=4, n_forced=0)
        ao = astro_object_from(message)

        det_mag = ao.detections[ao.detections["unit"] == "magnitude"]
        got = set(det_mag["candid"].tolist())
        for epoch in message["previous_detections"]:
            self.assertIn(epoch["measurement_id"], got)

    def test_forced_rows_keep_distnr_rfid_and_procstatus(self):
        message = generate_message(n_detections=4, n_previous_detections=2, n_forced=3)
        ao = astro_object_from(message)

        forced_mag = ao.forced_photometry[ao.forced_photometry["unit"] == "magnitude"]
        self.assertTrue(forced_mag["distnr"].notna().all())
        self.assertTrue(forced_mag["rfid"].notna().all())
        self.assertEqual({"0"}, set(forced_mag["procstatus"].unique()))

        det_mag = ao.detections[ao.detections["unit"] == "magnitude"]
        self.assertTrue(det_mag["rb"].notna().all())
        self.assertTrue(det_mag["distnr"].notna().all())

    def test_i_band_epochs_are_labelled_not_nan(self):
        # ZTF i-band is rare but real; the band map must keep its `i` entry so
        # those rows are labelled rather than becoming NaN.
        rng = random.Random(3)
        oid = 36028941624528297
        message = generate_message(oid=oid)
        message["detections"] = [
            candidate(oid, 1, "g", 60000.0, rng),
            candidate(oid, 2, "r", 60001.0, rng),
            candidate(oid, 3, "i", 60002.0, rng),
        ]
        message["previous_detections"] = []
        message["forced_photometries"] = []

        ao = astro_object_from(message)

        det_mag = ao.detections[ao.detections["unit"] == "magnitude"]
        self.assertEqual({"g", "r", "i"}, set(det_mag["fid"]))

    def test_forced_argument_must_be_empty(self):
        message = generate_message()
        prepared = build_step().pre_execute([message])[0]
        epochs = [
            {**e, "aid": e["oid"], "index_column": "x"} for e in prepared["detections"]
        ]

        with self.assertRaises(NotImplementedError):
            detections_to_astro_object(epochs, [epochs[0]], None, None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py::ParserTestCase -q -p no:warnings
```

Expected: `5 failed, 1 passed`. `test_forced_epochs_keep_their_corrected_magnitude` fails on
`brightness` being all-NaN for forced rows; `test_forced_argument_must_be_empty`
fails because no exception is raised.

- [ ] **Step 3: Drop the two rename entries from `DETECTION_KEYS_MAP`**

In `feature_step/features/utils/parsers.py`, replace:

```python
    "pid": "pid",
    "mag_corr":"magpsf_corr",
    "e_mag_corr_ext":"sigmapsf_corr_ext",
}
```

with:

```python
    "pid": "pid",
}
```

The LSST path selects neither key, so it is unaffected.

- [ ] **Step 4: Rewrite the head of `detections_to_astro_object`**

Replace everything from `detection_keys = [` through the
`a.fillna(value=np.nan, inplace=True)` line:

```python
    detection_keys = [
        "oid", #si
        "measurement_id", #si
        "aid", #placeholder
        "tid", # si
        "sid", # si
        "pid", #si 
        "ra", #si
        "dec", #si
        "mjd", #si
        "magpsf_corr",
        "sigmapsf_corr_ext",
        "mag", #si
        "e_mag", # si
        "band", #si
        "isdiffpos", #si
    ]

    values = []
    # Process regular detections (forced=False)
    for detection in detections:
        row = [detection.get(key, None) if key != 'sid' else str(detection.get(key, None)) for key in detection_keys]
        row.append(False)  # forced = False
        values.append(row)
    
    # Process forced photometry (forced=True)
    for detection in forced:
        row = [detection.get(key, None) if key != 'sid' else str(detection.get(key, None)) for key in detection_keys]
        row.append(True)  # forced = True
        values.append(row)

    a = pd.DataFrame(data=values, columns=detection_keys + ['forced'])
    a.fillna(value=np.nan, inplace=True)
```

with:

```python
    detection_keys = [
        "oid",
        "measurement_id",
        "aid",
        "tid",
        "sid",
        "pid",
        "ra",
        "dec",
        "mjd",
        "magpsf_corr",      # candidate / prv_candidate spelling
        "mag_corr",         # forced_photometry spelling
        "sigmapsf_corr_ext",
        "e_mag_corr_ext",
        "mag",
        "e_mag",
        "band",
        "isdiffpos",
        "forced",
    ]

    # Forced epochs arrive inline in `detections` with a per-row `forced` flag,
    # so this arg must be empty. A non-empty `forced` would misalign the
    # column-wise concat of the reference/bogus frames, which are computed over
    # `detections` alone.
    forced = forced or []
    if forced:
        raise NotImplementedError(
            "detections_to_astro_object: `forced` must be empty for ZTF; forced "
            "epochs flow inline via the per-row `forced` flag in `detections`."
        )

    values = []
    for detection in detections:
        row = [detection.get(key, None) if key != 'sid' else str(detection.get(key, None)) for key in detection_keys]
        values.append(row)

    a = pd.DataFrame(data=values, columns=detection_keys)
    a.fillna(value=np.nan, inplace=True)

    # Each row populates exactly one spelling of the corrected magnitude, so the
    # coalesce is unambiguous. It must run before the DETECTION_KEYS_MAP rename.
    a["magpsf_corr"] = a["magpsf_corr"].fillna(a["mag_corr"])
    a["sigmapsf_corr_ext"] = a["sigmapsf_corr_ext"].fillna(a["e_mag_corr_ext"])
    a.drop(columns=["mag_corr", "e_mag_corr_ext"], inplace=True)
```

Leave the rest of the function alone — `aid_forced = a[a["forced"]]` /
`aid_detections = a[~a["forced"]]` still do the split, now over a `forced` column
read straight from the row instead of appended per-loop.

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py -q -p no:warnings
```

Expected: `11 passed`.

- [ ] **Step 6: Confirm the legacy suite is unchanged**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf.py tests/unittest/test_step_lsst.py -q -p no:warnings
```

Expected: `8 failed, 4 passed`.

- [ ] **Step 7: Commit**

```bash
cd $REPO && git add feature_step/features/utils/parsers.py feature_step/tests/unittest/test_step_ztf_multisurvey.py
git commit -m "fix(feature_step): one loop over one epoch list in the ZTF parser

Fixes the concat misalignment that gave every forced row NaN distnr/rfid/
rb/procstatus, and coalesces mag_corr into magpsf_corr so forced epochs
keep their corrected magnitude."
```

---

## Task 5b: `procstatus` survives the trip through the bogus-flag frame

**Added during execution**, after the Task 2 code-quality review surfaced it. Not
in the original spec.

Task 2 made `discard_bogus_detections` tolerate an int `procstatus`. That fixes
the **first** call site (`pre_execute`, on raw message dicts). It does not fix the
**second** one: `ZTFLightcurvePreprocessor.drop_bogus_detections` re-runs the same
filter on `astro_object.forced_photometry.to_dict("records")`, and by then the
value has been through `get_bogus_flags_for_each_detection`, which does:

```python
bogus_flags = pd.DataFrame(bogus_flags, columns=keys)
bogus_flags["procstatus"] = bogus_flags["procstatus"].astype(str)
```

Detections carry no `procstatus` and forced rows do, so the column is always
`[None, ..., 0, ...]`. pandas types that as `float64`, and `.astype(str)` then
renders `0` as `"0.0"` — which matches neither `"0"` nor `"57"`, so **every forced
epoch is discarded**. Measured end-to-end: 6 surviving forced rows with
`procstatus="0"`, **0** with `procstatus=0`.

**Files:**
- Modify: `feature_step/features/utils/parsers.py` (`get_bogus_flags_for_each_detection`)
- Modify: `feature_step/tests/unittest/test_step_ztf_multisurvey.py`

- [ ] **Step 1: Write the failing test**

Append to `ParserTestCase` in
`feature_step/tests/unittest/test_step_ztf_multisurvey.py`:

```python
    def test_int_procstatus_survives_the_bogus_flag_frame(self):
        # `procstatus` is re-checked by the preprocessor after passing through a
        # DataFrame. A column mixing ints with None becomes float64, and 0 would
        # stringify to "0.0" — dropping every forced epoch.
        rng = random.Random(4)
        oid = 36028941624528297
        message = generate_message(oid=oid)
        message["detections"] = [
            candidate(oid, i, "g", 60000.0 + i, rng) for i in range(1, 6)
        ]
        message["previous_detections"] = []
        message["forced_photometries"] = [
            forced_photometry(oid, 100 + i, "g", 60010.0 + i, rng, procstatus=0)
            for i in range(3)
        ]

        ao = astro_object_from(message)
        self.assertEqual({"0"}, set(ao.forced_photometry["procstatus"]))

        ZTFLightcurvePreprocessor(drop_bogus=True).preprocess_single_object(ao)
        self.assertEqual(6, len(ao.forced_photometry))

    def test_int_procstatus_outside_the_allowed_set_is_still_dropped(self):
        rng = random.Random(5)
        oid = 36028941624528297
        message = generate_message(oid=oid)
        message["detections"] = [
            candidate(oid, i, "g", 60000.0 + i, rng) for i in range(1, 6)
        ]
        message["previous_detections"] = []
        message["forced_photometries"] = [
            forced_photometry(oid, 100 + i, "g", 60010.0 + i, rng, procstatus=2)
            for i in range(3)
        ]

        ao = astro_object_from(message)
        self.assertEqual(0, len(ao.forced_photometry))
```

The second test guards the fix from over-reaching: a genuinely bogus `procstatus`
must still be dropped, whether it arrives as `2` or `"2"`. (It is dropped in
`pre_execute`, before the frame is built, which is why the assertion is `0` rows
straight out of the parser.)

- [ ] **Step 2: Run the tests to verify the first one fails**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py -q -p no:warnings -k procstatus
```

Expected: `1 failed, 3 passed` — `test_int_procstatus_survives_the_bogus_flag_frame`
fails on `{'0.0'} != {'0'}`.

- [ ] **Step 3: Stringify from the original values**

In `feature_step/features/utils/parsers.py`, inside
`get_bogus_flags_for_each_detection`, replace:

```python
    bogus_flags = pd.DataFrame(bogus_flags, columns=keys)
    bogus_flags["procstatus"] = bogus_flags["procstatus"].astype(str)

    return bogus_flags
```

with:

```python
    # Stringify from the original values, not from the built column: procstatus
    # may arrive as an int, and a column mixing ints with None becomes float64,
    # so .astype(str) would render 0 as "0.0" and discard every forced epoch.
    procstatus = [str(row[keys.index("procstatus")]) for row in bogus_flags]

    bogus_flags = pd.DataFrame(bogus_flags, columns=keys)
    bogus_flags["procstatus"] = procstatus

    return bogus_flags
```

This preserves today's behaviour exactly for string input (a missing `procstatus`
still renders `"None"`, as it does now) and only changes the int case.

- [ ] **Step 4: Run the whole module to verify**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py -q -p no:warnings
```

Expected: `13 passed`.

- [ ] **Step 5: Confirm the legacy suite is unchanged**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf.py tests/unittest/test_step_lsst.py -q -p no:warnings
```

Expected: `8 failed, 4 passed`.

- [ ] **Step 6: Commit**

```bash
cd $REPO && git add feature_step/features/utils/parsers.py feature_step/tests/unittest/test_step_ztf_multisurvey.py
git commit -m "fix(feature_step): keep int procstatus intact through the bogus-flag frame

A column mixing ints with None types as float64, so .astype(str) rendered
procstatus 0 as \"0.0\" and the preprocessor discarded every forced epoch."
```

---

## Task 6: `execute` stamps `aid` and passes `forced=[]`

`add_mag_and_flux_columns` does `a.set_index("aid")` and the parser reads
`aid = a.index.values[0]`, but no ZTF multisurvey record has an `aid` field —
today the whole index is `NaN` and so is `metadata["aid"]`. The offline harness
sets `aid = oid` (`lc_features.py:30`); match it.

**Files:**
- Modify: `feature_step/features/step.py:368-371`
- Modify: `feature_step/tests/unittest/test_step_ztf_multisurvey.py`

- [ ] **Step 1: Write the failing tests**

Append to `feature_step/tests/unittest/test_step_ztf_multisurvey.py`, before the
`if __name__ == "__main__":` block:

```python
class ExecuteTestCase(unittest.TestCase):
    def test_execute_stamps_aid_and_passes_no_separate_forced_list(self):
        step = build_step()
        spy = mock.MagicMock(wraps=step.detections_to_astro_object_fn)
        step.detections_to_astro_object_fn = spy

        message = generate_message(n_detections=4, n_previous_detections=2, n_forced=3)
        step.execute(step.pre_execute([message]))

        passed_detections, passed_forced = spy.call_args[0][0], spy.call_args[0][1]
        self.assertEqual([], passed_forced)
        self.assertEqual(9, len(passed_detections))
        for epoch in passed_detections:
            self.assertEqual(epoch["oid"], epoch["aid"])
            self.assertEqual(
                f'{epoch["measurement_id"]}_{epoch["oid"]}', epoch["index_column"]
            )

    def test_execute_produces_features_and_scribe_commands(self):
        step = build_step()
        step.feature_name_lut = {0: "Amplitude", 1: "Multiband_period"}
        messages = [
            generate_message(
                oid=36028941624528297 + i,
                seed=42 + i,
                n_detections=30,
                n_previous_detections=20,
                n_forced=20,
                with_xmatch=True,
            )
            for i in range(2)
        ]

        results = step.execute(step.pre_execute(messages))

        self.assertEqual(2, len(results))
        for message, result in zip(messages, results):
            self.assertEqual(message["oid"], result["oid"])
            self.assertTrue(len(result["features"]) > 0)
        step.scribe_producer.producer.produce.assert_called()
```

Also add this to `ParserTestCase` (it belongs with the other metadata assertions):

```python
    def test_aid_is_the_oid(self):
        message = generate_message()
        ao = astro_object_from(message)

        aid = query_ao_table(ao.metadata, "aid")
        self.assertFalse(pd.isna(aid))
        self.assertEqual(message["oid"], aid)
        self.assertEqual([message["oid"]], list(ao.detections.index.unique()))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py::ExecuteTestCase -q -p no:warnings
```

Expected: `1 failed, 1 passed` —
`test_execute_stamps_aid_and_passes_no_separate_forced_list` fails with
`NotImplementedError` (Task 5's guard fires, because `execute` still passes the
`forced_photometries` array).

`test_aid_is_the_oid` passes already, because `astro_object_from` stamps `aid`
itself. That is intentional: it pins the parser-side contract while
`ExecuteTestCase` pins the step-side one.

- [ ] **Step 3: Implement the stamp**

In `feature_step/features/step.py`, inside `execute`, replace:

```python
            if self.survey == "ztf":
                forced = message.get("forced_photometries", None) #filtrar forced photometry
                xmatch_data = message.get("xmatches", None)
                ao = self.detections_to_astro_object_fn(list(m), forced ,xmatch_data, references_db)
```

with:

```python
            if self.survey == "ztf":
                # No ZTF multisurvey record has an `aid`; the parser indexes on it.
                m = map(lambda x: {**x, "aid": x["oid"]}, m)
                xmatch_data = message.get("xmatches", None)
                ao = self.detections_to_astro_object_fn(list(m), [], xmatch_data, references_db)
```

Forced epochs are already inside `message["detections"]` after Task 4, so there is
no separate list to pass.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py -q -p no:warnings
```

Expected: `16 passed`.

- [ ] **Step 5: Confirm the legacy suite is unchanged**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf.py tests/unittest/test_step_lsst.py -q -p no:warnings
```

Expected: `8 failed, 4 passed`.

- [ ] **Step 6: Commit**

```bash
cd $REPO && git add feature_step/features/step.py feature_step/tests/unittest/test_step_ztf_multisurvey.py
git commit -m "fix(feature_step): stamp aid=oid and pass forced=[] in the ZTF execute arm"
```

---

## Task 7: Fix the xmatch gate

`parsers.py:300` reads `if xmatches is not None and "allwise" in xmatches.keys():`,
but `step.py:302-311` attaches the `conesearch_with_metadata` result, whose keys
are `{oid, catalog, distance, match_id, metadata}`. The condition never fires, so
**W1–W4 are always NaN in the current ZTF path**.

This is hard-blocking, not cosmetic: `multisurvey_ztf.xmatch` and
`multisurvey_ztf.allwise` are both empty, so the live Xwave call is the only
source of WISE magnitudes. Eleven of the 127 seeded features depend on it.

**Files:**
- Modify: `feature_step/features/utils/parsers.py:300`
- Modify: `feature_step/tests/unittest/test_step_ztf_multisurvey.py`

- [ ] **Step 1: Write the failing tests**

Append to `ParserTestCase` in
`feature_step/tests/unittest/test_step_ztf_multisurvey.py`:

```python
    def test_wise_magnitudes_come_from_the_allwise_match(self):
        message = generate_message()
        ao = astro_object_from(message, xmatches=allwise_match(message["oid"]))

        self.assertEqual(15.1, query_ao_table(ao.metadata, "W1"))
        self.assertEqual(14.9, query_ao_table(ao.metadata, "W2"))
        self.assertEqual(12.5, query_ao_table(ao.metadata, "W3"))
        self.assertEqual(9.1, query_ao_table(ao.metadata, "W4"))

    def test_wise_magnitudes_are_nan_without_a_match(self):
        message = generate_message()
        ao = astro_object_from(message, xmatches=None)

        for name in ("W1", "W2", "W3", "W4"):
            self.assertTrue(pd.isna(query_ao_table(ao.metadata, name)))

    def test_wise_magnitudes_are_nan_for_another_catalog(self):
        message = generate_message()
        other = allwise_match(message["oid"])
        other["catalog"] = "gaia"
        ao = astro_object_from(message, xmatches=other)

        self.assertTrue(pd.isna(query_ao_table(ao.metadata, "W1")))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py::ParserTestCase -q -p no:warnings
```

Expected: `1 failed, 11 passed` — `test_wise_magnitudes_come_from_the_allwise_match` gets
`nan` instead of `15.1`. The two NaN tests pass already (for the wrong reason —
they pin the behavior so the fix cannot over-reach).

- [ ] **Step 3: Fix the gate**

In `feature_step/features/utils/parsers.py`, inside
`detections_to_astro_object`, replace:

```python
    if xmatches is not None and "allwise" in xmatches.keys(): #tentativo, a revisar
```

with:

```python
    if xmatches is not None and xmatches.get("catalog") == "allwise":
```

This matches the LSST parser at `parsers.py:108`.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py -q -p no:warnings
```

Expected: `19 passed`.

- [ ] **Step 5: Commit**

```bash
cd $REPO && git add feature_step/features/utils/parsers.py feature_step/tests/unittest/test_step_ztf_multisurvey.py
git commit -m "fix(feature_step): read W1-W4 from the conesearch match shape

The gate tested for an 'allwise' key that the conesearch result never has,
so W1-W4 were always NaN and the eleven WISE-derived features with them."
```

---

## Task 8: LUT-driven feature ids, and keep `oid` exact

Two independent defects in the same call chain.

**#3 — feature ids.** `prepare_ao_features_for_db` builds ids with `enumerate()`
over whatever names happen to be present, so the *same* feature gets a *different*
id depending on the batch. Take the LUT as an argument and invert it, exactly as
`prepare_ao_features_for_db_lsst` already does.

**Not in the spec — `oid` loses precision.** `detections_to_astro_object` builds
`metadata` from a list of `[name, value]` pairs and calls `.fillna(np.nan)`. Every
value is numeric, so the column becomes `float64` — and ZTF multisurvey oids
(~3.6e16) exceed `2**53`. `int(query_ao_table(metadata, "oid"))` currently returns
`36028941624528296` for oid `36028941624528297`, so **every scribe command would
carry a corrupted oid**. `parse_output`'s `assert oid_ao == oid` does not catch it
(the int is promoted to float for the comparison). LSST is unaffected — its
metadata frame contains the string `"aid"`, which keeps the column `object`.
Verified end-to-end: with the fix below, all 127 feature values are bit-identical
and `int(oid)` round-trips exactly.

**Files:**
- Modify: `feature_step/features/utils/parsers.py:335-352` (metadata frame)
- Modify: `feature_step/features/utils/parsers.py:408-441` (`prepare_ao_features_for_db`)
- Modify: `feature_step/features/utils/parsers.py:504` (call site)
- Modify: `feature_step/tests/unittest/test_step_ztf_multisurvey.py`

- [ ] **Step 1: Write the failing tests**

Append to `ParserTestCase`:

```python
    def test_oid_keeps_full_precision(self):
        # ZTF multisurvey oids exceed 2**53; a float64 metadata column rounds them.
        oid = 36028941624528297
        message = generate_message(oid=oid)
        ao = astro_object_from(message)

        self.assertEqual(oid, int(query_ao_table(ao.metadata, "oid")))
        self.assertEqual(oid, int(query_ao_table(ao.metadata, "aid")))
```

And append a new class before the `if __name__ == "__main__":` block:

```python
class FeatureIdTestCase(unittest.TestCase):
    def _astro_object_with_features(self, names):
        features = pd.DataFrame(
            [[name, 1.0 + i, None] for i, name in enumerate(names)],
            columns=["name", "value", "fid"],
        )
        ao = mock.MagicMock()
        ao.features = features
        return ao

    def test_feature_ids_come_from_the_injected_lut(self):
        lut = {7: "Amplitude", 11: "Beyond1Std", 42: "Multiband_period"}
        ao = self._astro_object_with_features(
            ["Amplitude", "Beyond1Std", "Multiband_period"]
        )

        result = prepare_ao_features_for_db(ao, lut)

        self.assertEqual(
            [7, 11, 42], result.sort_values("feature_id")["feature_id"].tolist()
        )

    def test_ids_are_stable_across_batches_with_different_feature_sets(self):
        lut = {7: "Amplitude", 11: "Beyond1Std", 42: "Multiband_period"}

        first = prepare_ao_features_for_db(
            self._astro_object_with_features(["Amplitude", "Multiband_period"]), lut
        )
        second = prepare_ao_features_for_db(
            self._astro_object_with_features(["Beyond1Std", "Multiband_period"]), lut
        )

        def id_of(frame, name):
            return frame[frame["name"] == name]["feature_id"].iloc[0]

        self.assertEqual(id_of(first, "Multiband_period"), id_of(second, "Multiband_period"))
        self.assertEqual(42, id_of(first, "Multiband_period"))

    def test_unknown_feature_name_maps_to_nan_and_warns(self):
        lut = {7: "Amplitude"}
        ao = self._astro_object_with_features(["Amplitude", "NotInTheLut"])

        with self.assertLogs("alerce.FeatureStep", level=logging.WARNING) as logs:
            result = prepare_ao_features_for_db(ao, lut)

        unknown = result[result["name"] == "NotInTheLut"]["feature_id"].iloc[0]
        self.assertTrue(pd.isna(unknown))
        self.assertIn("NotInTheLut", "".join(logs.output))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py -q -p no:warnings
```

Expected: `4 failed, 19 passed`. The three `FeatureIdTestCase` tests fail with
`TypeError: prepare_ao_features_for_db() takes 1 positional argument but 2 were given`;
`test_oid_keeps_full_precision` fails with `36028941624528296 != 36028941624528297`.

- [ ] **Step 3: Keep `oid`/`aid` exact in the metadata frame**

In `feature_step/features/utils/parsers.py`, inside
`detections_to_astro_object`, replace:

```python
    metadata = pd.DataFrame(
        [
            ["aid", aid],
            ["oid", oid],
            ["W1", w1],
            ["W2", w2],
            ["W3", w3],
            ["W4", w4],
            ["sgscore1", sgscore1],
            ["sgmag1", sgmag1],
            ["srmag1", srmag1],
            ["simag1", simag1],
            ["szmag1", szmag1],
            ["distpsnr1", distpsnr1],
            ["last_mjd", last_mjd],
        ],
        columns=["name", "value"],
    ).fillna(value=np.nan)
```

with:

```python
    metadata_rows = [
        ["aid", aid],
        ["oid", oid],
        ["W1", w1],
        ["W2", w2],
        ["W3", w3],
        ["W4", w4],
        ["sgscore1", sgscore1],
        ["sgmag1", sgmag1],
        ["srmag1", srmag1],
        ["simag1", simag1],
        ["szmag1", szmag1],
        ["distpsnr1", distpsnr1],
        ["last_mjd", last_mjd],
    ]
    # dtype=object keeps `oid`/`aid` as exact ints: ZTF multisurvey oids exceed
    # 2**53, so a float64 column silently rounds them. None -> NaN is done here
    # rather than with .fillna, which would downcast the column back to float64.
    metadata = pd.DataFrame(
        [[name, np.nan if value is None else value] for name, value in metadata_rows],
        columns=["name", "value"],
        dtype=object,
    )
```

- [ ] **Step 4: Take the LUT as an argument**

Replace the signature:

```python
def prepare_ao_features_for_db(astro_object: AstroObject) -> pd.DataFrame: #esto tengo que verlo
```

with:

```python
def prepare_ao_features_for_db(astro_object: AstroObject, feature_name_lut) -> pd.DataFrame:
```

and, inside the same function, replace:

```python
    #deberia usar el feature_name_lut para mapear los nombres a ids,
    unique_feature_names = ao_features["name"].unique()
    name_to_id = {name: idx for idx, name in enumerate(unique_feature_names)}
    #print(name_to_id)
```

with:

```python
    # Invert the LUT so ids are stable across batches, regardless of which
    # feature names happen to be present in this one.
    name_to_id = {name: feature_id for feature_id, name in feature_name_lut.items()}
```

- [ ] **Step 5: Update the call site**

In `parse_scribe_payload`, replace:

```python
        ao_features = prepare_ao_features_for_db(astro_object)
```

with:

```python
        ao_features = prepare_ao_features_for_db(astro_object, feature_name_lut)
```

`parse_scribe_payload` already receives `feature_name_lut` as its fourth
parameter. It is the only caller in this repo (verified with grep).

- [ ] **Step 6: Run the tests to verify they pass**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py -q -p no:warnings
```

Expected: `23 passed`.

- [ ] **Step 7: Confirm the legacy suite is unchanged**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf.py tests/unittest/test_step_lsst.py -q -p no:warnings
```

Expected: `8 failed, 4 passed`.

- [ ] **Step 8: Commit**

```bash
cd $REPO && git add feature_step/features/utils/parsers.py feature_step/tests/unittest/test_step_ztf_multisurvey.py
git commit -m "fix(feature_step): stable ZTF feature ids from the LUT, exact oids

feature_id came from enumerate() over the batch, so the same feature got a
different id per batch. The metadata frame was float64, which rounded ZTF
oids past 2**53 and would have written the wrong oid to every scribe command."
```

---

## Task 9: End-to-end parity — the step emits all 127 seeded feature names

Ties Task 3 (the extractor port) to the rest of the pipeline. The deployed
`multisurvey_ztf.feature_name_lut` has 127 rows, byte-identical to
`features/offline/feature_lut.py::FEATURE_NAME_LUT`. This is the assertion that
the step emits every one of them.

**Files:**
- Modify: `feature_step/tests/unittest/test_step_ztf_multisurvey.py`

- [ ] **Step 1: Write the test**

Append to `feature_step/tests/unittest/test_step_ztf_multisurvey.py`, before the
`if __name__ == "__main__":` block:

```python
class ExtractorParityTestCase(unittest.TestCase):
    """The deployed feature_name_lut has 127 rows; the step must emit all of them."""

    def test_extractor_emits_the_127_seeded_feature_names(self):
        message = generate_message(
            n_detections=30, n_previous_detections=20, n_forced=20
        )
        ao = astro_object_from(message, xmatches=allwise_match(message["oid"]))
        ZTFLightcurvePreprocessor(drop_bogus=True).preprocess_single_object(ao)
        ZTFFeatureExtractor().compute_features_single_object(ao)

        names = set(ao.features["name"].unique())
        self.assertEqual(127, len(names))
        for name in ("SPM_mjd_ref", "TDE_mjd_ref", "fleet_mjd_ref", "ulens_mjd_ref"):
            self.assertIn(name, names)
```

- [ ] **Step 2: Run the test to verify it passes**

Run:

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py::ExtractorParityTestCase -q -p no:warnings
```

Expected: `1 passed`. Task 3 already landed the extractor changes, so this test
is green on arrival — it exists to lock the parity in.

If it reports `123 != 127`, Task 3's copy did not take effect: check `PYTHONPATH`
resolves `lc_classifier` to this repo (the sanity check at the top of this plan).

- [ ] **Step 3: Commit**

```bash
cd $REPO && git add feature_step/tests/unittest/test_step_ztf_multisurvey.py
git commit -m "test(feature_step): assert the ZTF step emits all 127 seeded feature names"
```

---

## Task 10: Final verification

- [ ] **Step 1: Run the full new suite**

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest/test_step_ztf_multisurvey.py -q -p no:warnings
```

Expected: `24 passed`.

- [ ] **Step 2: Run the whole unit suite and compare to baseline**

```bash
cd $REPO/feature_step && $PY -m pytest tests/unittest -q -p no:warnings
```

Expected: `8 failed, 28 passed` — the same 8 legacy failures as the baseline
(7 in `test_step_ztf.py`, 1 in `test_step_lsst.py`), plus the 24 new tests and the
4 legacy passes. **If the failure count or the failing test names differ from the
baseline, stop and report it.**

- [ ] **Step 3: Run the lc_classifier test**

```bash
cd $REPO/lc_classifier && $PY -m pytest tests/features/test_discard_bogus_detections.py -q -p no:warnings
```

Expected: `6 passed`.

- [ ] **Step 4: Confirm the diff touches only the intended files**

```bash
cd $REPO && git diff --stat main...HEAD
```

Expected exactly these files, and nothing else — the two `docs/` entries are the
spec and this plan, which are versioned alongside the change:

```
docs/superpowers/plans/2026-08-06-feature-step-ztf-multisurvey.md
docs/superpowers/specs/2026-08-04-feature-step-ztf-multisurvey-design.md
feature_step/features/step.py
feature_step/features/utils/parsers.py
feature_step/tests/message_factory_ztf_ms.py
feature_step/tests/unittest/test_step_ztf_multisurvey.py
lc_classifier/lc_classifier/features/core/base.py
lc_classifier/lc_classifier/features/extractors/spm_extractor.py
lc_classifier/lc_classifier/features/extractors/tde_extractor.py
lc_classifier/lc_classifier/features/extractors/ulens_extractor.py
lc_classifier/tests/features/test_discard_bogus_detections.py
```

- [ ] **Step 5: Report what is still needed before deployment**

These are out of scope per the spec, and each blocks deployment independently.
Restate them in the completion message so nobody mistakes a green suite for a
deployable step:

1. **Output schema.** `schemas/feature_step/output.avsc` is still legacy ZTF (`oid: string`, `candid: array<string>`, `fid: string`) and does not match what `parse_output` emits.
2. **Helm chart.** `charts/feature_step/values.yaml` consumes the `xmatch` topic and defines no `SURVEY`, `DB_CONFIG`, `USE_XMATCH`, `XMATCH_CONFIG` or `XMATCH_CATALOGS`.
3. **`settings.py` parity.** It never defines those keys either; only the `CONFIG_FROM_YAML` path works.
4. **Scribe.** `scribe_multisurvey/.../decode.py` has no `survey == "ztf" and step == "features"` route, so the command this step emits hits the `raise`. ZTF feature rows will not reach `multisurvey_ztf.feature` until that is handled.
5. **Deployment config.** `DB_CONFIG.SCHEMA` must be set to `multisurvey_ztf` (it defaults to `multisurvey`), and the step's DB user needs INSERT on `multisurvey_ztf.feature_version_lut` — `27.7.1` is not in the LUT, so first startup inserts it as `version_id=1`.
6. **Forward message shape.** After the flatten, `parse_output` emits the merged, bogus-filtered light curve under `"detections"` rather than the three original arrays. Whatever consumes the ZTF feature topic sees the change.
