# Offline ZTF Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the BHRF light-curve classifier (`SquidwardFeaturesClassifier`, v2.1.0) offline, end-to-end: `oid → DB → correction-ztf message → features → probabilities`.

**Architecture:** Extend the existing de-vendored offline tooling in `feature_step/features/offline/`. Reuse real pipeline code: `features.utils.parsers.parse_output` (feature_step) names the features, the real `alerce_classifiers.base.factories.input_dto_factory` builds a **features-only** `InputDTO`, and the real `SquidwardFeaturesClassifier` predicts. The Squidward model reads only `InputDTO.features`, so detections are passed empty — sidestepping the local `lc_classification_step`'s stale candid-based schema entirely. No `lc_classification` dependency.

**Tech Stack:** Python 3.10, pandas, poetry; `alerce_classifiers[ztf]` (git submodule), `lc_classifier`, `features` (feature_step). Tests with pytest.

**Reference:** design doc `docs/superpowers/specs/2026-06-21-offline-ztf-classification-design.md`.

---

## File Structure

- `feature_step/pyproject.toml` — add `alerce_classifiers[ztf]` path dep.
- `feature_step/features/offline/lc_features.py` — extract `compute_astro_object(...)` (returns the post-extract `AstroObject`); `compute_features` delegates to it.
- `feature_step/features/offline/classify.py` — **new.** Model loading + the message→DTO→prediction bridge.
- `feature_step/features/offline/db.py` — add deferred `fetch_stored_probabilities(...)` stub.
- `feature_step/scripts/offline_classify.py` — **new.** End-to-end CLI for one oid.
- `feature_step/scripts/offline_compare_probabilities.py` — **new.** Deferred skeleton.
- `feature_step/tests/unittest/test_offline_classify.py` — **new.** Unit tests for the bridge.
- `feature_step/features/offline/README.md` — document classification + scripts.

---

## Task 0: Prerequisite — submodule (already done during planning)

**Files:** none (environment).

- [ ] **Step 1: Verify the `alerce_classifiers` submodule is checked out**

Run: `ls /home/fandrades/desktop/pipeline/alerce_classifiers/pyproject.toml`
Expected: the path exists (file listed). If missing, run:
`git -C /home/fandrades/desktop/pipeline submodule update --init alerce_classifiers`

---

## Task 1: Add `alerce_classifiers[ztf]` to the feature_step env

**Files:**
- Modify: `feature_step/pyproject.toml:13-18` (dependencies block)

- [ ] **Step 1: Add the path dependency**

In `feature_step/pyproject.toml`, under `[tool.poetry.dependencies]`, after the `lc-classifier` line (line 13), add:

```toml
alerce_classifiers = { path = "../alerce_classifiers", develop = true, extras = ["ztf"] }
```

- [ ] **Step 2: Install it into the working environment**

Run (from `feature_step/`): `poetry lock --no-update && poetry install`

If the project is run via the existing conda env instead of poetry (the offline scripts use `conda run -n training_py310`), install editable into that env instead:
Run: `pip install -e "/home/fandrades/desktop/pipeline/alerce_classifiers[ztf]"`

Expected: install completes without dependency-resolution errors.

- [ ] **Step 3: Verify the imports resolve**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && python -c "from alerce_classifiers.base.factories import input_dto_factory; from alerce_classifiers.squidward.model import SquidwardFeaturesClassifier; from alerce_classifiers.squidward.mapper import SquidwardMapper; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
cd /home/fandrades/desktop/pipeline
git add feature_step/pyproject.toml feature_step/poetry.lock
git commit -m "build(feature_step): add alerce_classifiers[ztf] for offline classification"
```

---

## Task 2: Extract `compute_astro_object` in `lc_features.py`

Pure refactor: the classify path needs the post-extract `AstroObject` (not just `ao.features`). Extract a function that returns the `AstroObject`; `compute_features` keeps its current return for back-compat.

**Files:**
- Modify: `feature_step/features/offline/lc_features.py:59-76` (`compute_features`)

- [ ] **Step 1: Replace `compute_features` with an extracted `compute_astro_object` + delegating `compute_features`**

Replace the existing `compute_features` function (lines 59-76) with:

```python
def compute_astro_object(message: dict, references_db, allwise, min_detections: int = 1,
                         preprocessor=None, extractor=None):
    """Per-oid path: message -> AstroObject -> preprocess -> extract -> AstroObject.

    `preprocessor`/`extractor` are injectable for tests; defaults are the production
    stack, constructed lazily (the extractor is heavy). Returns the post-extract
    AstroObject, or None if the message has too few real detections."""
    if preprocessor is None:
        preprocessor = ZTFLightcurvePreprocessor(drop_bogus=True)
    if extractor is None:
        extractor = ZTFFeatureExtractor()

    ao = message_to_astro_object(message, references_db, allwise, min_detections)
    if ao is None:
        return None
    preprocessor.preprocess_single_object(ao)
    extractor.compute_features_single_object(ao)
    return ao


def compute_features(message: dict, references_db, allwise, min_detections: int = 1,
                     preprocessor=None, extractor=None):
    """Per-oid path: message -> AstroObject -> preprocess -> extract -> features.

    Returns the long features frame, or None if the message has too few real
    detections."""
    ao = compute_astro_object(message, references_db, allwise, min_detections,
                              preprocessor=preprocessor, extractor=extractor)
    if ao is None:
        return None
    return ao.features
```

- [ ] **Step 2: Verify existing offline tests still pass (no regression)**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && python -m pytest tests/unittest/test_offline_feature_compare.py -v`
Expected: PASS (these don't touch `compute_features`, but confirm the module still imports cleanly).

- [ ] **Step 3: Verify the module imports**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && python -c "from features.offline.lc_features import compute_astro_object, compute_features; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
cd /home/fandrades/desktop/pipeline
git add feature_step/features/offline/lc_features.py
git commit -m "refactor(feature_step): extract compute_astro_object for offline reuse"
```

---

## Task 3: The classification bridge — `classify.py` (TDD)

**Files:**
- Create: `feature_step/features/offline/classify.py`
- Test: `feature_step/tests/unittest/test_offline_classify.py`

- [ ] **Step 1: Write the failing tests**

Create `feature_step/tests/unittest/test_offline_classify.py`:

```python
"""Tests for features/offline/classify.py — the message->DTO->prediction bridge.

The real SquidwardFeaturesClassifier is replaced by fakes; parse_output is
monkeypatched so no real AstroObject/DB is needed. Requires alerce_classifiers
installed (for OutputDTO / input_dto_factory)."""
import pandas as pd
import pytest

from alerce_classifiers.base.dto import OutputDTO
from features.offline import classify


def test_features_message_to_dto_builds_wide_frame():
    out_message = {"oid": 123, "features": {"Amplitude_1": 0.5, "Period_2": 2.0}}
    dto = classify.features_message_to_dto(out_message)
    feats = dto.features
    assert list(feats.index) == [123]
    assert feats.loc[123, "Amplitude_1"] == 0.5
    assert feats.loc[123, "Period_2"] == 2.0
    assert dto.detections.empty


def test_features_message_to_dto_handles_missing_features():
    out_message = {"oid": 9, "features": None}
    dto = classify.features_message_to_dto(out_message)
    assert list(dto.features.index) == [9]


class _FakeModel:
    """Stub model: records the DTO it received and returns canned probabilities."""
    def __init__(self):
        self.received = None

    def can_predict(self, dto):
        return True, ""

    def predict(self, dto):
        self.received = dto
        return OutputDTO(
            pd.DataFrame({"AGN": [0.7]}, index=[123]),
            {"top": pd.DataFrame(), "children": {}},
        )


class _CantModel:
    def can_predict(self, dto):
        return False, "Empty features found"

    def predict(self, dto):
        raise AssertionError("predict must not be called when can_predict is False")


def test_classify_astro_object_predicts(monkeypatch):
    out_message = {"oid": 123, "measurement_id": [1], "features": {"Amplitude_1": 0.5}}
    monkeypatch.setattr(classify, "parse_output", lambda aos, msgs, candids: [out_message])
    model = _FakeModel()

    result = classify.classify_astro_object(object(), {"oid": 123, "measurement_id": [1]}, model)

    assert result.probabilities.loc[123, "AGN"] == 0.7
    assert model.received.features.loc[123, "Amplitude_1"] == 0.5


def test_classify_astro_object_cant_predict_returns_empty(monkeypatch):
    out_message = {"oid": 1, "measurement_id": [], "features": {}}
    monkeypatch.setattr(classify, "parse_output", lambda aos, msgs, candids: [out_message])

    result = classify.classify_astro_object(object(), {"oid": 1, "measurement_id": []}, _CantModel())

    assert result.probabilities.empty


def test_load_squidward_model_requires_model_path(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with pytest.raises(ValueError, match="MODEL_PATH"):
        classify.load_squidward_model()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && python -m pytest tests/unittest/test_offline_classify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'features.offline.classify'`.

- [ ] **Step 3: Write `classify.py`**

Create `feature_step/features/offline/classify.py`:

```python
"""Offline classification: features AstroObject -> BHRF probabilities.

Stitches the real feature_step output parser (`parse_output`), the real
alerce_classifiers InputDTO factory, and the real SquidwardFeaturesClassifier
(BHRF) together, without the lc_classification Kafka step. The Squidward model
reads only InputDTO.features, so detections/non_detections/xmatch/stamps are
passed empty (this also sidesteps lc_classification_step's stale candid schema).

Model config is read from the same env vars the deployed step uses:
    MODEL_PATH   (required) - model pickle URL/path (e.g. the S3 BHRF 2.1.0 url)
    MAPPER_CLASS (optional) - defaults to the Squidward mapper
    CLASSIFIER_NAME (optional) - output label; deployment uses lc_classifier_BHRF_forced_phot
"""
import os

import pandas as pd
from apf.core import get_class
from alerce_classifiers.base.dto import OutputDTO
from alerce_classifiers.base.factories import input_dto_factory

from features.utils.parsers import parse_output
from .lc_features import compute_astro_object

DEFAULT_MODEL_CLASS = "alerce_classifiers.squidward.model.SquidwardFeaturesClassifier"
DEFAULT_MAPPER_CLASS = "alerce_classifiers.squidward.mapper.SquidwardMapper"


def load_squidward_model(model_class: str = DEFAULT_MODEL_CLASS):
    """Instantiate the BHRF classifier from env vars (mirrors the deployed step).

    Returns (model, classifier_name, classifier_version). The version is derived
    by the model from the model path (e.g. ".../squidward/2.1.0/..." -> "2.1.0").
    """
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        raise ValueError("MODEL_PATH env var is required to load the model")
    mapper_class = os.getenv("MAPPER_CLASS", DEFAULT_MAPPER_CLASS)
    mapper = get_class(mapper_class)()
    model = get_class(model_class)(model_path=model_path, mapper=mapper)
    name = os.getenv("CLASSIFIER_NAME", model_class.split(".")[-1])
    return model, name, model.model_version


def features_message_to_dto(out_message: dict):
    """Classifier-input message -> features-only InputDTO.

    `out_message["features"]` is the wide, band-suffixed feature dict produced by
    parse_output. The model reads only features; detections/non_detections/xmatch/
    stamps are empty."""
    features = out_message.get("features") or {}
    features_df = pd.DataFrame([features], index=[out_message["oid"]])
    features_df.index.name = "oid"
    empty = pd.DataFrame()
    return input_dto_factory(empty, empty, features_df, empty, empty)


def _empty_output() -> OutputDTO:
    return OutputDTO(pd.DataFrame(), {"top": pd.DataFrame(), "children": {}})


def classify_astro_object(ao, message: dict, model) -> OutputDTO:
    """Post-extract AstroObject + its source message -> OutputDTO.

    Uses the real parse_output to name the features, builds a features-only DTO,
    then runs can_predict + predict. Returns an empty OutputDTO if the model
    can't predict (mirrors the step's behavior)."""
    candids = {message["oid"]: message.get("measurement_id", [])}
    out_message = parse_output([ao], [message], candids)[0]
    dto = features_message_to_dto(out_message)
    can, _ = model.can_predict(dto)
    if not can:
        return _empty_output()
    return model.predict(dto)


def classify_oid(oid: int, credentials: str, model, min_detections: int = 1,
                 preprocessor=None, extractor=None):
    """DB -> message -> features -> probabilities for one oid.

    Returns an OutputDTO, or None if the object has too few real detections."""
    from features.offline import db
    from features.offline.message import build_message

    oids = [oid]
    dets = db.fetch_detections(credentials, oids)
    forced = db.fetch_forced_photometry(credentials, oids)
    ps1 = db.fetch_ps1(credentials, oids)
    allwise = db.fetch_allwise(credentials, oids)
    refs = db.fetch_references(credentials, oids)

    message = build_message(oid, dets, forced, ps1)
    ao = compute_astro_object(message, refs, allwise, min_detections,
                              preprocessor=preprocessor, extractor=extractor)
    if ao is None:
        return None
    return classify_astro_object(ao, message, model)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && python -m pytest tests/unittest/test_offline_classify.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/fandrades/desktop/pipeline
git add feature_step/features/offline/classify.py feature_step/tests/unittest/test_offline_classify.py
git commit -m "feat(feature_step): offline BHRF classification bridge (features-only DTO)"
```

---

## Task 4: End-to-end CLI — `offline_classify.py`

**Files:**
- Create: `feature_step/scripts/offline_classify.py`

- [ ] **Step 1: Write the script**

Create `feature_step/scripts/offline_classify.py`:

```python
#!/usr/bin/env python
"""Live-DB check: DB -> message -> features -> BHRF probabilities for one ZTF oid.

Requires MODEL_PATH (and optionally MAPPER_CLASS) env vars, same as the deployed step:

    MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl \
        conda run --no-capture-output -n training_py310 python \
        feature_step/scripts/offline_classify.py --oid 36028941624528297
"""
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]   # .../pipeline
for p in (PIPE / "feature_step", PIPE / "lc_classifier",
          PIPE / "libs" / "idmapper", PIPE / "alerce_classifiers"):
    sys.path.insert(0, str(p))

import argparse

from features.offline.classify import load_squidward_model, classify_oid

DEFAULT_CREDENTIALS = "/home/fandrades/desktop/repos/training/features_ztf/data/credentials.json"


def main():
    ap = argparse.ArgumentParser(
        description="Offline DB->features->BHRF probabilities for one ZTF oid."
    )
    ap.add_argument("--oid", type=int, required=True,
                    help="Multisurvey bigint oid (e.g. 36028941624528297).")
    ap.add_argument("--credentials", default=DEFAULT_CREDENTIALS,
                    help="Path to DB credentials JSON.")
    ap.add_argument("--min-det", type=int, default=1,
                    help="Minimum real detections required to classify.")
    args = ap.parse_args()

    model, name, version = load_squidward_model()
    print(f"model: {name} version={version}")
    print(f"oid: {args.oid}")

    result = classify_oid(args.oid, args.credentials, model, min_detections=args.min_det)
    if result is None or result.probabilities is None or len(result.probabilities) == 0:
        print("\nFAIL: no probabilities (too few detections or can't predict)")
        sys.exit(1)

    print(f"\nprobabilities:\n{result.probabilities.to_string()}")
    top = result.hierarchical.get("top")
    if top is not None and len(top):
        print(f"\ntop:\n{top.to_string()}")
    print("\nOK: probabilities produced.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses args and imports cleanly (no DB/model yet)**

Run: `cd /home/fandrades/desktop/pipeline && python feature_step/scripts/offline_classify.py --help`
Expected: prints the argparse help (usage with `--oid`, `--credentials`, `--min-det`); exit 0.

- [ ] **Step 3: Commit**

```bash
cd /home/fandrades/desktop/pipeline
git add feature_step/scripts/offline_classify.py
git commit -m "feat(feature_step): offline_classify.py end-to-end BHRF CLI"
```

---

## Task 5: Deferred compare — DB stub + script skeleton

**Files:**
- Modify: `feature_step/features/offline/db.py` (append at end of file)
- Create: `feature_step/scripts/offline_compare_probabilities.py`

- [ ] **Step 1: Add the deferred DB reader stub**

Append to `feature_step/features/offline/db.py`:

```python
def fetch_stored_probabilities(credentials, oids):
    """Read stored BHRF probabilities, for compare-vs-offline. DEFERRED.

    The stored-probability table (schema/name and the classifier_name/version
    filter) is not yet pinned down. Wire this up once it is; see the offline
    classification design doc."""
    raise NotImplementedError(
        "pending: stored-probability table TBD "
        "(see docs/superpowers/specs/2026-06-21-offline-ztf-classification-design.md)"
    )
```

- [ ] **Step 2: Create the deferred compare script skeleton**

Create `feature_step/scripts/offline_compare_probabilities.py`:

```python
#!/usr/bin/env python
"""Compare offline BHRF probabilities vs. stored probabilities. DEFERRED skeleton.

Pending the stored-probability table definition. Once db.fetch_stored_probabilities
is implemented, this will mirror offline_compare_vs_alerce.py for probabilities:
run the offline pipeline on an oid, fetch the stored probabilities, and diff.
"""
import sys


def main():
    print(
        "offline_compare_probabilities is not implemented yet: pending the "
        "stored-probability table definition (see the offline classification design doc)."
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the stub raises and the skeleton exits as designed**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && python -c "from features.offline import db; \
import sys; \
exec('try:\n db.fetch_stored_probabilities(None, [1])\nexcept NotImplementedError as e:\n print(\"raised:\", str(e)[:20])')"`
Expected: prints `raised: pending: stored-pro`.

Run: `cd /home/fandrades/desktop/pipeline && python feature_step/scripts/offline_compare_probabilities.py; echo "exit=$?"`
Expected: prints the "not implemented yet" message and `exit=2`.

- [ ] **Step 4: Commit**

```bash
cd /home/fandrades/desktop/pipeline
git add feature_step/features/offline/db.py feature_step/scripts/offline_compare_probabilities.py
git commit -m "feat(feature_step): deferred stored-probability compare stub + skeleton"
```

---

## Task 6: Documentation + manual end-to-end verification

**Files:**
- Modify: `feature_step/features/offline/README.md`

- [ ] **Step 1: Add a Classification section to the README**

In `feature_step/features/offline/README.md`, add a row to the Modules table for `classify.py`:

```
| `classify.py` | Classification bridge. `load_squidward_model()` builds the BHRF `SquidwardFeaturesClassifier` from env vars (`MODEL_PATH`, `MAPPER_CLASS`); `classify_astro_object(ao, message, model)` names features via the real `parse_output`, builds a **features-only** `InputDTO` (`input_dto_factory`), and runs `can_predict`+`predict`; `classify_oid(...)` is the DB->probabilities convenience path. The Squidward model reads only `features`, so detections are passed empty (no `lc_classification` dependency). |
```

And add rows to the Scripts table:

```
| `offline_classify.py --oid <bigint> [--credentials PATH] [--min-det M]` | DB -> message -> features -> BHRF probabilities for one oid. Requires `MODEL_PATH` env (the S3 BHRF url). |
| `offline_compare_probabilities.py` | **Deferred** — predicted vs. stored probabilities. Pending the stored-probability table. |
```

- [ ] **Step 2: Add a short "Classification" prose subsection**

After the "Running" section in the README, add:

```markdown
## Classification (BHRF)

`classify.py` runs the deployed **BHRF** model (`SquidwardFeaturesClassifier`,
`lc_classifier_BHRF_forced_phot`, v2.1.0) on offline-computed features. Model
config comes from the same env vars the step uses (`MODEL_PATH`, `MAPPER_CLASS`).

```bash
cd feature_step
MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl \
    poetry run python scripts/offline_classify.py --oid 36028941624528297 --credentials /path/to/credentials.json
```

The model is **features-only**: it reads only `InputDTO.features`, so the bridge
builds the DTO with empty detections via the real `alerce_classifiers` factory and
does not depend on `lc_classification` (whose local schema is candid-based and
incompatible with v11 messages).
```

- [ ] **Step 3: Run the full unit test suite for the offline module**

Run: `cd /home/fandrades/desktop/pipeline/feature_step && python -m pytest tests/unittest/test_offline_classify.py tests/unittest/test_offline_feature_compare.py -v`
Expected: all tests PASS.

- [ ] **Step 4: Manual end-to-end run on a real oid**

Run (with a valid credentials path and network access to S3):
```bash
cd /home/fandrades/desktop/pipeline
MODEL_PATH=https://alerce-models.s3.amazonaws.com/squidward/2.1.0/hierarchical_random_forest_model.pkl \
    python feature_step/scripts/offline_classify.py --oid 36028941624528297
```
Expected: prints `model: ... version=2.1.0`, a populated `probabilities` frame indexed by oid, and `OK: probabilities produced.`

If it fails on feature-name mismatch (model `feature_list` vs `parse_output` names), capture the diff between `result`/`features_df.columns` and `model.model.feature_list` and report — that indicates a feature-naming gap to reconcile, not a bridge bug.

- [ ] **Step 5: Commit**

```bash
cd /home/fandrades/desktop/pipeline
git add feature_step/features/offline/README.md
git commit -m "docs(feature_step): document offline BHRF classification"
```

---

## Self-Review notes

- **Spec coverage:** model loading from env vars (Task 3 `load_squidward_model`, B); end-to-end oid→probabilities (Task 3 `classify_oid` + Task 4 script, C); features-only DTO via real factory (Task 3); deferred compare (Task 5); location A / `alerce_classifiers[ztf]`-only dep (Task 1); README (Task 6). All spec sections map to a task.
- **Type consistency:** `load_squidward_model` → `(model, name, version)`; `features_message_to_dto(out_message)` → `InputDTO`; `classify_astro_object(ao, message, model)` → `OutputDTO`; `classify_oid(...)` → `OutputDTO | None`. Names used consistently across tasks and tests.
- **No placeholders:** every code/test/command step is concrete. The only intentional "pending" is the deferred compare (Task 5), which is explicit and tested for its NotImplementedError/exit-2 behavior.
