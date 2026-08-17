# Multisurvey LC Classification Step Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `lc_classification_multisurvey_step/`, a standalone apf step that consumes the multisurvey `feature_step` output topic, runs the ZTF BHRF (Squidward 2.1.0) classifier, and produces five heads of probabilities to `scribe_multisurvey`, with classifier ids and the class taxonomy both resolved from the database at startup.

**Architecture:** A copy-and-adapt sibling of `stamp_classifier_2025_multisurvey_step/`, porting the already-validated offline logic in `~/desktop/pipeline/feature_step/features/offline/`. The step owns no model code — the model and mapper are resolved from config via `apf.core.get_class`. Persistence is scribe-only; the step never writes to the DB, it only reads `classifier` and `taxonomy` once at startup. The pure logic (`probabilities.py`, `db/db.py`, `output_parser.py`, and the frame-building half of `input_dto.py`) deliberately does **not** import `alerce_classifiers`, so the whole unit suite runs without the model dependency; only `step.py` and the opt-in integration test need it.

**Tech Stack:** Python 3.10, pandas 2.2, SQLAlchemy 2.0 (Core `text()` only), apf-base, db-plugins-multisurvey, alerce_classifiers (git submodule), pytest.

**Spec:** `docs/superpowers/specs/2026-08-16-multisurvey-lc-classification-step-design.md`. Every task below cites the spec sections it implements. If the code and the spec disagree, that is a finding to raise, not a thing to silently paper over.

---

## Environment (read this before Task 1)

There is **no poetry** on this machine and the step's own virtualenv is not created as part of this plan. Every test command in this plan uses the pre-existing conda env that already has every runtime dependency:

```
PYTHON=/home/fandrades/miniconda3/envs/feature_step/bin/python
```

It provides pandas 2.2.2, pytest, sqlalchemy 2.0.19, `apf`, and `db_plugins.db.sql.models_pipeline`. `pyproject.toml` is still written (Task 1) because it is the deploy artifact, but nothing in this plan runs `poetry install`.

The `alerce_classifiers` submodule has been initialised at the repo root (`git submodule update --init alerce_classifiers`, pinned at `3316a3e`). Its importable package is the **inner** directory, so anything that imports it needs:

```
PYTHONPATH=/home/fandrades/desktop/pipeline_features/pipeline/alerce_classifiers
```

Only Task 9 (`step.py`) and Task 11 (integration test) need that. Tasks 2-8 must pass **without** it — that is an explicit design constraint, and Task 2's test asserts it.

All commands are run from the step directory unless stated otherwise:

```
cd /home/fandrades/desktop/pipeline_features/pipeline/lc_classification_multisurvey_step
```

Work happens on the current branch `feat/multisurvey-lc-classification-step`. Do not create a worktree.

### Two environment traps, both verified

Neither is a code defect and neither affects the container build, but both will
mislead you if you validate locally without knowing about them.

**1. The importable `apf` is not the one this step depends on.** `pyproject.toml`
declares `apf-base = { path = "../libs/apf", develop = true }`, but the
`feature_step` conda env resolves `apf` to a *different project tree*:

```
$ python -c "import apf, apf.core.step as s; print(apf.__file__, hasattr(s.GenericStep,'_flush_producers'))"
/home/fandrades/desktop/online/pipeline/libs/apf/apf/__init__.py False
```

That stale copy has **no `_flush_producers` and no `KafkaProducer.flush`**; the
in-repo `libs/apf` has both (`apf/core/step.py:302,318`). Any local check of apf
behaviour — producer flushing, offset commits, the `start()` loop — is therefore
checking the wrong code unless you force the in-repo copy first on `PYTHONPATH`.
State which `apf` a result came from whenever it matters. Before trusting Task
11, `poetry install` against the in-repo path.

**2. `numexpr` is not installed in any conda env on this machine.** It is
correctly declared (`pyproject.toml`) and imported identically by all four
sibling classifier steps, so the image is fine. Locally, either install it or
put a throwaway stub on `PYTHONPATH` outside the repo — the stub needs
`__version__` as well as `utils.set_num_threads`, because pandas probes the
version. Never remove the import to make a check pass.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Deploy artifact: deps, `step` entrypoint. Not installed by this plan. |
| `settings.py` | env → config dict. Consumer/scribe/producer/metrics/PSQL/model blocks. |
| `models_settings.py` | One-entry `configurator` returning the `MODEL_CONFIG` block. |
| `scripts/run_step.py` | Entrypoint: yaml-or-settings config, logging, prometheus, `step.start()`. |
| `lc_classification_multisurvey_step/probabilities.py` | **Pure.** Head names/frames, version→smallint, `OutputDTO` → scribe row dicts. No alerce, no DB. |
| `lc_classification_multisurvey_step/db/db.py` | `PSQLConnection`, the two read-only queries, and `resolve_classifiers` (the §8 startup assertions). No pandas, no alerce. |
| `lc_classification_multisurvey_step/input_dto.py` | Message filtering, features frame, `lastmjd` map, and a lazily-importing DTO factory wrapper. |
| `lc_classification_multisurvey_step/output_parser.py` | **PLACEHOLDER** downstream producer payload (spec §9). Duck-typed, no alerce. |
| `lc_classification_multisurvey_step/step.py` | The only module that imports apf + alerce. Wires everything; owns no logic worth unit-testing. |
| `tests/unittest/test_probabilities.py` | Tasks 2, 3. |
| `tests/unittest/test_taxonomy.py` | Tasks 4, 5. |
| `tests/unittest/test_input_dto.py` | Task 6. |
| `tests/unittest/test_output_parser.py` | Task 7. |
| `tests/integration/test_offline_equivalence.py` | Task 11, opt-in. |

Shared vocabulary, fixed here so tasks do not drift:

- **head** — one of the five BHRF outputs. Identified by a **classifier name**, never by a hardcoded id.
- `classifier_ids` — `{classifier_name: classifier_id}`, from the DB.
- `taxonomy_maps` — `{classifier_id: {class_name: class_id}}`, from the DB.
- `lastmjd_map` — `{oid: float}`.

---

### Task 1: Scaffold the step package

Implements: spec §3 (module layout).

**Files:**
- Create: `lc_classification_multisurvey_step/pyproject.toml`
- Create: `lc_classification_multisurvey_step/README.md`
- Create: `lc_classification_multisurvey_step/.gitignore`
- Create: `lc_classification_multisurvey_step/lc_classification_multisurvey_step/__init__.py`
- Create: `lc_classification_multisurvey_step/lc_classification_multisurvey_step/db/__init__.py`
- Create: `lc_classification_multisurvey_step/tests/__init__.py`
- Create: `lc_classification_multisurvey_step/tests/unittest/__init__.py`
- Test: `lc_classification_multisurvey_step/tests/unittest/test_package_imports.py`

- [ ] **Step 1: Create the directory tree and empty package markers**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline
mkdir -p lc_classification_multisurvey_step/lc_classification_multisurvey_step/db
mkdir -p lc_classification_multisurvey_step/scripts
mkdir -p lc_classification_multisurvey_step/tests/unittest
mkdir -p lc_classification_multisurvey_step/tests/integration
touch lc_classification_multisurvey_step/lc_classification_multisurvey_step/__init__.py
touch lc_classification_multisurvey_step/lc_classification_multisurvey_step/db/__init__.py
touch lc_classification_multisurvey_step/tests/__init__.py
touch lc_classification_multisurvey_step/tests/unittest/__init__.py
touch lc_classification_multisurvey_step/tests/integration/__init__.py
```

`tests/__init__.py` and `tests/unittest/__init__.py` are load-bearing: they make pytest insert the step directory (not `tests/`) onto `sys.path`, which is how `import lc_classification_multisurvey_step` resolves without any install. This mirrors `stamp_classifier_2025_multisurvey_step/tests/__init__.py`.

- [ ] **Step 2: Write the failing test**

Create `lc_classification_multisurvey_step/tests/unittest/test_package_imports.py`:

```python
"""The package layout itself is a contract: the unit suite must import the pure
modules with no alerce_classifiers and no apf on the path (spec: the model lives
in the submodule, the step is the only module that needs it)."""
import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "lc_classification_multisurvey_step",
        "lc_classification_multisurvey_step.db",
    ],
)
def test_package_modules_import(module):
    assert importlib.import_module(module) is not None
```

- [ ] **Step 3: Run test to verify it passes**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline/lc_classification_multisurvey_step
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest -v
```

Expected: `2 passed`. If it errors with `ModuleNotFoundError: lc_classification_multisurvey_step`, the `__init__.py` files from Step 1 are missing or you are not in the step directory.

(This test passes on creation rather than failing first — it asserts the scaffolding exists, and there is no meaningful red state for `mkdir`. Every subsequent task is strict red-green.)

- [ ] **Step 4: Write `pyproject.toml`**

Create `lc_classification_multisurvey_step/pyproject.toml`. Deps mirror `stamp_classifier_2025_multisurvey_step/pyproject.toml`, minus the stamp-only ones (`astropy`, `idmapper`, `wget`) and with the `ztf` extra of `alerce_classifiers` — that is the extra carrying `imbalanced-learn`, which `alerce_classifiers.classifiers.hierarchical_random_forest` imports (`from imblearn.ensemble import BalancedRandomForestClassifier`). There is no `squidward` extra.

```toml
[tool.poetry]
name = "lc-classification-multisurvey-step"
version = "0.1.0"
description = "Multisurvey LC classification step (ZTF BHRF / Squidward)"
authors = []
readme = "README.md"
packages = [{include = "lc_classification_multisurvey_step"}]

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.poetry.scripts]
step = { callable = "scripts.run_step:step" }

[tool.poetry.dependencies]
python = ">=3.10,<3.12"
pandas = "^2.0.1"
numexpr = "^2.8.8"
apf-base = { path = "../libs/apf", develop = true }
db-plugins = { path = "../libs/db-plugins-multisurvey", develop = true }
alerce_classifiers = { path = "../alerce_classifiers", develop = true, extras = ["ztf"] }
sqlalchemy = "^2.0.19"
psycopg2-binary = "^2.9.6"

[tool.poetry.group.test.dependencies]
pytest = "^7.2.0"
pytest-cov = "^4.0.0"

[tool.poetry.group.dev.dependencies]
black = "~=23.0"

[tool.black]
line-length = 100
```

- [ ] **Step 5: Write `.gitignore` and `README.md`**

`lc_classification_multisurvey_step/.gitignore`:

```
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/
*.egg-info/
config.yaml
config.yml
config.*.yaml
```

The `config.*` rules match every sibling step (`correction_multisurvey_step`, `magstats_multisurvey_step`, `feature_step`, and the two stamp steps all ignore their local config). `scripts/run_step.py` reads `/config/config.yaml` when `CONFIG_FROM_YAML` is set, so a developer running this step locally will have a credential-bearing `config.yaml` in this directory.

`lc_classification_multisurvey_step/README.md`:

```markdown
# LC Classification Multisurvey Step

Consumes the multisurvey `feature_step` output topic, runs the ZTF BHRF
(Squidward 2.1.0) classifier, and produces probabilities for its five heads to
`scribe_multisurvey`, which owns the upsert into `multisurvey_ztf.probability`.

The step writes nothing to the database. It reads `classifier` and `taxonomy`
once at startup to resolve classifier names to ids and class names to class ids,
and refuses to start if either is unseeded (see the design doc, §8).

Design: `docs/superpowers/specs/2026-08-16-multisurvey-lc-classification-step-design.md`

## Tests

The unit suite has no model dependency:

    python -m pytest tests/unittest -v

The offline-equivalence test is opt-in and needs the `alerce_classifiers`
submodule plus `MODEL_PATH`:

    RUN_EQUIVALENCE_TEST=1 MODEL_PATH=<s3 url> python -m pytest tests/integration -v
```

- [ ] **Step 6: Re-run the test and commit**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline/lc_classification_multisurvey_step
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest -v
cd /home/fandrades/desktop/pipeline_features/pipeline
git add lc_classification_multisurvey_step
git commit -m "feat(lc_classification_multisurvey_step): scaffold package and test harness"
```

Expected: `2 passed`, then a commit.

---

### Task 2: Head names and version→smallint

Implements: spec §6 (the five heads, name-derived), §7 (`classifier_version`).

**Files:**
- Create: `lc_classification_multisurvey_step/lc_classification_multisurvey_step/probabilities.py`
- Test: `lc_classification_multisurvey_step/tests/unittest/test_probabilities.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unittest/test_probabilities.py`:

```python
"""Pure-function tests for the probability row builder.

Deliberately imports no alerce_classifiers: `probabilities.py` is duck-typed over
anything with `.probabilities` and `.hierarchical`, so a namespace stub stands in
for the real OutputDTO.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from lc_classification_multisurvey_step import probabilities as p


def make_dto(flat=None, top=None, transient=None, stochastic=None, periodic=None):
    """Stand-in for alerce_classifiers OutputDTO (probabilities + hierarchical)."""
    children = {}
    if transient is not None:
        children["Transient"] = transient
    if stochastic is not None:
        children["Stochastic"] = stochastic
    if periodic is not None:
        children["Periodic"] = periodic
    return SimpleNamespace(
        probabilities=flat if flat is not None else pd.DataFrame(),
        hierarchical={"top": top, "children": children},
    )


def frame(index, data):
    """{class_name: [values]} -> DataFrame indexed by oid, like the model emits."""
    df = pd.DataFrame(data, index=index)
    df.index.name = "oid"
    return df


class TestHeadNames:
    def test_default_base_name_matches_the_seeded_classifier(self):
        assert p.DEFAULT_CLASSIFIER_NAME == "lc_classifier_BHRF_forced_phot"

    def test_five_heads_in_flat_top_transient_stochastic_periodic_order(self):
        assert p.head_names("base") == [
            "base",
            "base_top",
            "base_transient",
            "base_stochastic",
            "base_periodic",
        ]

    def test_defaults_to_the_deployed_base_name(self):
        assert p.head_names()[0] == "lc_classifier_BHRF_forced_phot"
        assert p.head_names()[4] == "lc_classifier_BHRF_forced_phot_periodic"


class TestClassifierVersionToSmallint:
    def test_three_part_version(self):
        assert p.classifier_version_to_smallint("2.1.0") == 210

    def test_strips_suffix_on_the_patch_part(self):
        assert p.classifier_version_to_smallint("2.1.0_rc1") == 210

    def test_non_three_part_version_is_zero(self):
        assert p.classifier_version_to_smallint("dev") == 0
        assert p.classifier_version_to_smallint("2.1") == 0


class TestIterHeadFrames:
    def test_pairs_each_head_name_with_its_frame(self):
        flat = frame([1], {"SNIa": [0.9]})
        top = frame([1], {"Transient": [0.8]})
        transient = frame([1], {"SNIa": [0.7]})
        stochastic = frame([1], {"AGN": [0.6]})
        periodic = frame([1], {"LPV": [0.5]})
        dto = make_dto(flat, top, transient, stochastic, periodic)

        got = p.iter_head_frames(dto, "base")

        assert [name for name, _ in got] == p.head_names("base")
        assert got[0][1] is flat
        assert got[1][1] is top
        assert got[2][1] is transient
        assert got[3][1] is stochastic
        assert got[4][1] is periodic

    def test_missing_children_yield_none_rather_than_raising(self):
        dto = make_dto(flat=frame([1], {"SNIa": [0.9]}), top=None)
        got = dict(p.iter_head_frames(dto, "base"))
        assert got["base_top"] is None
        assert got["base_transient"] is None

    def test_absent_hierarchical_yields_none_for_the_four_hierarchical_heads(self):
        dto = SimpleNamespace(probabilities=frame([1], {"SNIa": [0.9]}), hierarchical=None)
        got = dict(p.iter_head_frames(dto, "base"))
        assert got["base"] is not None
        assert all(got[n] is None for n in p.head_names("base")[1:])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline/lc_classification_multisurvey_step
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest/test_probabilities.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'lc_classification_multisurvey_step.probabilities'`.

- [ ] **Step 3: Write minimal implementation**

Create `lc_classification_multisurvey_step/probabilities.py`:

```python
"""BHRF OutputDTO -> scribe-ready probability rows.

Ported from the offline reference `features/offline/probability_writer.py`, with
two deliberate changes (design doc §5):

  - the offline builder is strictly per-oid and raises on a multi-row frame; this
    one is batched, so it melts by oid;
  - offline pins `CLASSIFIER_IDS = [5..9]`; here the ids come from the database
    and only the head *names* are pinned (design doc §6).

Pure: no database, no alerce_classifiers, no apf. `output_dto` is duck-typed —
anything with `.probabilities` and `.hierarchical` works.
"""
import logging

log = logging.getLogger(__name__)

DEFAULT_CLASSIFIER_NAME = "lc_classifier_BHRF_forced_phot"
DEFAULT_CLASSIFIER_VERSION = "2.1.0"

# Positional against the model's hierarchical output; pinned, not configurable.
HEAD_SUFFIXES = ("", "_top", "_transient", "_stochastic", "_periodic")


def head_names(base_name: str = DEFAULT_CLASSIFIER_NAME) -> list:
    """The five classifier names for `base_name`, in head order."""
    return [f"{base_name}{suffix}" for suffix in HEAD_SUFFIXES]


def classifier_version_to_smallint(version: str) -> int:
    """'2.1.0' -> 210. Strips a '_suffix' on the patch part. 0 if not 3 parts."""
    parts = version.split(".")
    if len(parts) == 3:
        parts[-1] = parts[-1].split("_")[0]
        return int("".join(parts))
    return 0


def iter_head_frames(output_dto, base_name: str = DEFAULT_CLASSIFIER_NAME) -> list:
    """[(classifier_name, frame_or_None)] for the five heads, in head order."""
    hierarchical = getattr(output_dto, "hierarchical", None) or {}
    children = hierarchical.get("children") or {}
    names = head_names(base_name)
    return [
        (names[0], output_dto.probabilities),
        (names[1], hierarchical.get("top")),
        (names[2], children.get("Transient")),
        (names[3], children.get("Stochastic")),
        (names[4], children.get("Periodic")),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest/test_probabilities.py -v
```

Expected: `9 passed`.

- [ ] **Step 5: Verify the no-alerce constraint holds**

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -c "
import sys
from lc_classification_multisurvey_step import probabilities
assert not [m for m in sys.modules if m.startswith('alerce_classifiers')], 'probabilities.py pulled in alerce_classifiers'
print('OK: probabilities.py imports no alerce_classifiers')
"
```

Expected: `OK: probabilities.py imports no alerce_classifiers`.

- [ ] **Step 6: Commit**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline
git add lc_classification_multisurvey_step
git commit -m "feat(lc_classification_multisurvey_step): head names and version smallint"
```

---

### Task 3: `build_probability_rows`

Implements: spec §5 (batched melt), §6 (heads), §7 (row contract, per-head dense ranking), §8 (unknown class → skip that oid's rows for that head and log).

**Files:**
- Modify: `lc_classification_multisurvey_step/lc_classification_multisurvey_step/probabilities.py`
- Test: `lc_classification_multisurvey_step/tests/unittest/test_probabilities.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unittest/test_probabilities.py`:

```python
# --- build_probability_rows ------------------------------------------------

NAMES = p.head_names("base")
IDS = {NAMES[0]: 50, NAMES[1]: 60, NAMES[2]: 70, NAMES[3]: 80, NAMES[4]: 90}
TAXONOMY = {
    50: {"SNIa": 0, "AGN": 1, "LPV": 2},
    60: {"Transient": 0, "Stochastic": 1, "Periodic": 2},
    70: {"SNIa": 0, "SLSN": 1},
    80: {"AGN": 0, "QSO": 1},
    90: {"LPV": 0, "EA": 1},
}
# Ids are deliberately NOT 5-9: a reintroduced hardcode must fail these tests.


def build(dto, lastmjd_map, ids=IDS, taxonomy=TAXONOMY, **kw):
    return p.build_probability_rows(
        dto, lastmjd_map, ids, taxonomy, base_name="base", version="2.1.0", **kw
    )


class TestBuildProbabilityRows:
    def test_empty_output_dto_yields_no_rows(self):
        assert build(make_dto(), {}) == []

    def test_none_output_dto_yields_no_rows(self):
        assert build(None, {}) == []

    def test_flat_head_row_contract(self):
        dto = make_dto(flat=frame([123], {"SNIa": [0.7], "AGN": [0.2], "LPV": [0.1]}))

        rows = build(dto, {123: 60000.5})

        assert len(rows) == 3
        by_class = {r["class_id"]: r for r in rows}
        assert by_class[0] == {
            "oid": 123,
            "sid": 0,
            "classifier_id": 50,
            "classifier_version": 210,
            "class_id": 0,
            "probability": pytest.approx(0.7),
            "ranking": 1,
            "lastmjd": 60000.5,
        }
        assert set(rows[0]) == {
            "oid", "sid", "classifier_id", "classifier_version",
            "class_id", "probability", "ranking", "lastmjd",
        }

    def test_sid_is_configurable(self):
        dto = make_dto(flat=frame([123], {"SNIa": [1.0]}))
        rows = build(dto, {123: 1.0}, sid=3)
        assert {r["sid"] for r in rows} == {3}

    def test_melts_a_multi_oid_frame(self):
        dto = make_dto(flat=frame([1, 2], {"SNIa": [0.6, 0.1], "AGN": [0.4, 0.9]}))

        rows = build(dto, {1: 100.0, 2: 200.0})

        assert len(rows) == 4
        assert {r["oid"] for r in rows} == {1, 2}
        oid2 = [r for r in rows if r["oid"] == 2]
        assert {r["lastmjd"] for r in oid2} == {200.0}
        # ranking is per (oid, head): oid 2's AGN wins even though oid 1's SNIa is higher
        agn_id = TAXONOMY[50]["AGN"]
        assert [r["ranking"] for r in oid2 if r["class_id"] == agn_id] == [1]

    def test_ranking_is_dense_descending_within_oid_and_head(self):
        dto = make_dto(flat=frame([1], {"SNIa": [0.5], "AGN": [0.5], "LPV": [0.0]}))

        rows = build(dto, {1: 1.0})

        rank_by_class = {r["class_id"]: r["ranking"] for r in rows}
        assert rank_by_class[TAXONOMY[50]["SNIa"]] == 1
        assert rank_by_class[TAXONOMY[50]["AGN"]] == 1  # tie -> same dense rank
        assert rank_by_class[TAXONOMY[50]["LPV"]] == 2  # dense, not 3

    def test_all_five_heads_are_emitted(self):
        dto = make_dto(
            flat=frame([1], {"SNIa": [1.0]}),
            top=frame([1], {"Transient": [1.0]}),
            transient=frame([1], {"SNIa": [1.0]}),
            stochastic=frame([1], {"AGN": [1.0]}),
            periodic=frame([1], {"LPV": [1.0]}),
        )

        rows = build(dto, {1: 1.0})

        assert sorted(r["classifier_id"] for r in rows) == [50, 60, 70, 80, 90]

    def test_missing_and_empty_heads_are_skipped(self):
        dto = make_dto(
            flat=frame([1], {"SNIa": [1.0]}),
            top=None,
            transient=frame([], {"SNIa": []}),
        )

        rows = build(dto, {1: 1.0})

        assert {r["classifier_id"] for r in rows} == {50}

    def test_unknown_class_drops_the_whole_head(self, caplog):
        dto = make_dto(
            flat=frame([1, 2], {"SNIa": [0.5, 0.5], "Nonsense": [0.5, 0.5]}),
            top=frame([1, 2], {"Transient": [1.0, 1.0]}),
        )

        with caplog.at_level("ERROR"):
            rows = build(dto, {1: 1.0, 2: 2.0})

        # an unknown class name is frame-wide: the flat head is dropped entirely,
        # top survives
        assert {r["classifier_id"] for r in rows} == {60}
        assert len(rows) == 2
        assert "Nonsense" in caplog.text

    def test_known_classes_keep_every_oid(self):
        # Counterpart to the test above: with no unknown class name, nothing drops.
        dto = make_dto(flat=frame([1, 2], {"SNIa": [0.5, 0.5], "AGN": [0.5, 0.5]}))

        rows = build(dto, {1: 1.0, 2: 2.0})

        assert {r["oid"] for r in rows} == {1, 2}
        assert len(rows) == 4

    def test_oid_without_lastmjd_is_dropped_and_logged(self, caplog):
        dto = make_dto(flat=frame([1, 2], {"SNIa": [1.0, 1.0]}))

        with caplog.at_level("ERROR"):
            rows = build(dto, {1: 1.0})

        assert {r["oid"] for r in rows} == {1}
        assert "2" in caplog.text

    def test_head_with_no_taxonomy_map_is_dropped_and_logged(self, caplog):
        dto = make_dto(flat=frame([1], {"SNIa": [1.0]}), top=frame([1], {"Transient": [1.0]}))
        taxonomy = {50: TAXONOMY[50]}  # no map for the top head's id

        with caplog.at_level("ERROR"):
            rows = p.build_probability_rows(
                dto, {1: 1.0}, IDS, taxonomy, base_name="base", version="2.1.0"
            )

        assert {r["classifier_id"] for r in rows} == {50}
        assert "60" in caplog.text

    def test_head_with_no_resolved_id_is_dropped_and_logged(self, caplog):
        dto = make_dto(flat=frame([1], {"SNIa": [1.0]}), top=frame([1], {"Transient": [1.0]}))
        ids = {NAMES[0]: 50}  # top head never resolved

        with caplog.at_level("ERROR"):
            rows = p.build_probability_rows(
                dto, {1: 1.0}, ids, TAXONOMY, base_name="base", version="2.1.0"
            )

        assert {r["classifier_id"] for r in rows} == {50}
        assert NAMES[1] in caplog.text

    def test_values_are_native_python_types_for_json_serialisation(self):
        import json

        dto = make_dto(flat=frame([1], {"SNIa": [0.5], "AGN": [0.5]}))
        rows = build(dto, {1: 1.0})
        json.dumps(rows)  # numpy int64/float64 would raise here

    def test_multi_oid_multi_head_with_one_head_dropped(self):
        dto = make_dto(
            flat=frame([1, 2], {"SNIa": [0.9, 0.1], "AGN": [0.1, 0.9]}),
            top=frame([1, 2], {"Transient": [0.8, 0.2], "Stochastic": [0.2, 0.8]}),
            periodic=frame([1, 2], {"LPV": [0.6, 0.4], "Unseeded": [0.4, 0.6]}),
        )

        rows = build(dto, {1: 100.0, 2: 200.0})

        # flat (50) and top (60) survive for both oids; periodic (90) is dropped
        # entirely because "Unseeded" is not in its taxonomy.
        assert {r["classifier_id"] for r in rows} == {50, 60}
        assert len(rows) == 8
        assert {(r["oid"], r["classifier_id"]) for r in rows} == {
            (1, 50), (1, 60), (2, 50), (2, 60),
        }
        assert {r["lastmjd"] for r in rows if r["oid"] == 2} == {200.0}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest/test_probabilities.py -v
```

Expected: the `TestBuildProbabilityRows` tests fail with `AttributeError: module ... has no attribute 'build_probability_rows'`; the Task 2 tests still pass.

- [ ] **Step 3: Write minimal implementation**

Append to `lc_classification_multisurvey_step/probabilities.py`:

```python
def build_probability_rows(
    output_dto,
    lastmjd_map: dict,
    classifier_ids: dict,
    taxonomy_maps: dict,
    *,
    base_name: str = DEFAULT_CLASSIFIER_NAME,
    version: str = DEFAULT_CLASSIFIER_VERSION,
    sid: int = 0,
) -> list:
    """Batched BHRF OutputDTO -> scribe-ready probability row dicts (all 5 heads).

    Parameters
    ----------
    output_dto : anything with `.probabilities` and `.hierarchical`, or None.
        Caller contract: each head's frame must have a unique oid index. Duplicate
        oids are not collapsed here and would emit rows colliding on the probability
        primary key; de-duplication happens upstream when the features frame is built.
    lastmjd_map : {oid: lastmjd}. An oid missing from it is dropped and logged —
        `probability.lastmjd` is NOT NULL.
    classifier_ids : {classifier_name: classifier_id}, from the DB (design §6.1).
    taxonomy_maps : {classifier_id: {class_name: class_id}}, from the DB.

    Per design §8, problems detectable only per-batch are logged and drop the
    affected (oid, head) rows rather than killing the batch. Startup problems are
    the caller's job (`db.resolve_classifiers`).
    """
    if output_dto is None or output_dto.probabilities is None:
        return []

    version_smallint = classifier_version_to_smallint(version)
    rows = []

    for classifier_name, frame in iter_head_frames(output_dto, base_name):
        if frame is None or len(frame) == 0:
            continue

        classifier_id = classifier_ids.get(classifier_name)
        if classifier_id is None:
            log.error(
                "no classifier_id resolved for head '%s'; dropping %d rows for this head",
                classifier_name,
                len(frame),
            )
            continue

        class_id_of = taxonomy_maps.get(classifier_id)
        if not class_id_of:
            log.error(
                "no taxonomy map for classifier_id=%s (head '%s'); dropping this head",
                classifier_id,
                classifier_name,
            )
            continue

        # Class names are the frame's COLUMNS, so an unknown class name is a
        # frame-wide model/taxonomy drift, never a per-oid condition — check once
        # per head rather than once per oid.
        unknown = sorted(set(frame.columns) - set(class_id_of))
        if unknown:
            log.error(
                "classifier_id=%s (head '%s'): class names %s absent from the taxonomy; "
                "dropping this head for all %d oids in the batch",
                classifier_id,
                classifier_name,
                unknown,
                len(frame),
            )
            continue

        melted = (
            frame.rename_axis("oid")
            .reset_index()
            .melt(id_vars=["oid"], var_name="class_name", value_name="probability")
        )
        melted["ranking"] = (
            melted.groupby("oid")["probability"]
            .rank(ascending=False, method="dense")
            .astype(int)
        )
        melted["oid"] = melted["oid"].astype("int64")
        melted["lastmjd"] = melted["oid"].map(lastmjd_map)

        missing_lastmjd = melted["lastmjd"].isna()
        if missing_lastmjd.any():
            dropped = sorted(set(melted.loc[missing_lastmjd, "oid"]))
            log.error(
                "oids %s have no lastmjd; dropping their rows for classifier_id=%s",
                dropped,
                classifier_id,
            )
            melted = melted[~missing_lastmjd]
            if melted.empty:
                continue

        melted["class_id"] = melted["class_name"].map(class_id_of)

        for record in melted.to_dict("records"):
            rows.append(
                {
                    "oid": int(record["oid"]),
                    "sid": int(sid),
                    "classifier_id": int(classifier_id),
                    "classifier_version": int(version_smallint),
                    "class_id": int(record["class_id"]),
                    "probability": float(record["probability"]),
                    "ranking": int(record["ranking"]),
                    "lastmjd": float(record["lastmjd"]),
                }
            )

    return rows
```

**Why vectorised.** Ranking is computed before any row is dropped, so it always ranks over all classes of the head for that oid. `class_id` and `lastmjd` are resolved with `.map()` over the whole melted frame and `to_dict("records")` is called once per head, not once per (oid, head) — the latter costs ~3 s of pure CPU per 1000-oid batch (5,000 tiny `to_dict` calls), on a step that runs this for every Kafka batch. This is the shape the sibling `stamp_classifier_2025_multisurvey_step/.../db.py::format_probability_records` already uses. The explicit `int()`/`float()` casts stay: they are what keeps numpy scalars out of `json.dumps` in the produce path.

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest/test_probabilities.py -v
```

Expected: all pass (9 from Task 2 + 15 from Task 3 = `24 passed` in this file, `26 passed` for the whole unit suite).

- [ ] **Step 5: Commit**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline
git add lc_classification_multisurvey_step
git commit -m "feat(lc_classification_multisurvey_step): batched probability row builder"
```

---

### Task 4: DB readers

Implements: spec §6.1 (`get_classifier_ids_by_name`), §4 (startup reads), §5 (`fetch_taxonomy_maps` adapted).

**Files:**
- Create: `lc_classification_multisurvey_step/lc_classification_multisurvey_step/db/db.py`
- Test: `lc_classification_multisurvey_step/tests/unittest/test_taxonomy.py`

Note the deliberate difference from `stamp_classifier_2025_multisurvey_step/.../db/db.py`: that one wraps its query in `try/except` and returns `{}` on error, which makes a dead DB indistinguishable from an unseeded table. Here exceptions propagate (design §6.1).

- [ ] **Step 1: Write the failing test**

Create `tests/unittest/test_taxonomy.py`:

```python
"""Tests for the two read-only startup queries and the startup assertions.

The session is faked rather than mocked with MagicMock so the assertions read as
"given these DB rows, ...". No database is involved.
"""
from contextlib import contextmanager

import pytest

from lc_classification_multisurvey_step.db import db


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self._rows


class FakeSession:
    """Returns canned rows and records the statements it was asked to execute."""

    def __init__(self, rows_by_call):
        self._rows_by_call = list(rows_by_call)
        self.executed = []

    def execute(self, statement, params=None):
        self.executed.append((str(statement), params))
        return FakeResult(self._rows_by_call.pop(0))


class FakeConnection:
    def __init__(self, *rows_by_call):
        self.session_obj = FakeSession(rows_by_call)

    @contextmanager
    def session(self):
        yield self.session_obj


CLASSIFIER_ROWS = [
    {"classifier_id": 50, "classifier_name": "base", "classifier_version": "2.1.0"},
    {"classifier_id": 60, "classifier_name": "base_top", "classifier_version": "2.1.0"},
]


class TestGetClassifierIdsByName:
    def test_maps_name_to_id_and_version(self):
        conn = FakeConnection(CLASSIFIER_ROWS)

        got = db.get_classifier_ids_by_name(["base", "base_top"], conn)

        assert got == {
            "base": {"classifier_id": 50, "classifier_version": "2.1.0"},
            "base_top": {"classifier_id": 60, "classifier_version": "2.1.0"},
        }

    def test_ids_are_not_assumed_to_be_five_through_nine(self):
        conn = FakeConnection(
            [{"classifier_id": 41, "classifier_name": "base", "classifier_version": "2.1.0"}]
        )
        assert db.get_classifier_ids_by_name(["base"], conn)["base"]["classifier_id"] == 41

    def test_rows_returned_out_of_order_still_map_correctly(self):
        conn = FakeConnection(list(reversed(CLASSIFIER_ROWS)))
        got = db.get_classifier_ids_by_name(["base", "base_top"], conn)
        assert got["base"]["classifier_id"] == 50

    def test_names_are_passed_as_a_bound_parameter(self):
        conn = FakeConnection(CLASSIFIER_ROWS)
        db.get_classifier_ids_by_name(["base", "base_top"], conn)
        _statement, params = conn.session_obj.executed[0]
        assert params == {"names": ["base", "base_top"]}

    def test_missing_name_is_simply_absent(self):
        conn = FakeConnection([CLASSIFIER_ROWS[0]])
        got = db.get_classifier_ids_by_name(["base", "base_top"], conn)
        assert "base_top" not in got

    def test_duplicate_name_raises(self):
        conn = FakeConnection(
            [
                {"classifier_id": 50, "classifier_name": "base", "classifier_version": "2.1.0"},
                {"classifier_id": 51, "classifier_name": "base", "classifier_version": "2.1.0"},
            ]
        )
        with pytest.raises(ValueError, match="base"):
            db.get_classifier_ids_by_name(["base"], conn)

    def test_db_errors_propagate_rather_than_returning_an_empty_map(self):
        class Boom(FakeConnection):
            @contextmanager
            def session(self):
                raise RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            db.get_classifier_ids_by_name(["base"], Boom())


TAXONOMY_ROWS = [
    {"classifier_id": 50, "class_id": 0, "class_name": "SNIa"},
    {"classifier_id": 50, "class_id": 1, "class_name": "AGN"},
    {"classifier_id": 60, "class_id": 0, "class_name": "Transient"},
]


class TestGetTaxonomyByClassifierId:
    def test_groups_class_names_by_classifier_id(self):
        conn = FakeConnection(TAXONOMY_ROWS)

        got = db.get_taxonomy_by_classifier_id([50, 60], conn)

        assert got == {50: {"SNIa": 0, "AGN": 1}, 60: {"Transient": 0}}

    def test_ids_are_passed_as_a_bound_parameter(self):
        conn = FakeConnection(TAXONOMY_ROWS)
        db.get_taxonomy_by_classifier_id([50, 60], conn)
        _statement, params = conn.session_obj.executed[0]
        assert params == {"classifier_ids": [50, 60]}

    def test_classifier_with_no_rows_is_absent(self):
        conn = FakeConnection([TAXONOMY_ROWS[0]])
        assert db.get_taxonomy_by_classifier_id([50, 60], conn) == {50: {"SNIa": 0}}

    def test_db_errors_propagate(self):
        class Boom(FakeConnection):
            @contextmanager
            def session(self):
                raise RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            db.get_taxonomy_by_classifier_id([50], Boom())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest/test_taxonomy.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'lc_classification_multisurvey_step.db.db'`.

- [ ] **Step 3: Write minimal implementation**

Create `lc_classification_multisurvey_step/db/db.py`:

```python
"""Read-only database access for the multisurvey LC classification step.

Two queries, both run once at startup: classifier names -> ids, and classifier
ids -> {class_name: class_id}. The step never writes to the database; the scribe
owns the probability upsert (design doc §2, decision 3).

Unlike stamp_classifier_2025_multisurvey_step's reader, neither query swallows
exceptions: an unreachable database must not look like an unseeded table.
"""
import logging
from contextlib import contextmanager
from typing import Callable, ContextManager

from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

log = logging.getLogger(__name__)


def get_db_url(config: dict) -> str:
    return (
        f"postgresql://{config['USER']}:{config['PASSWORD']}"
        f"@{config['HOST']}:{config['PORT']}/{config['DB_NAME']}"
    )


class PSQLConnection:
    """Session factory over a psql engine, scoped to `SCHEMA` via search_path.

    Copied from stamp_classifier_2025_multisurvey_step/.../db/db.py so the two
    multisurvey classifier steps connect identically.
    """

    def __init__(self, db_config: dict, engine=None, poolclass: str | None = None) -> None:
        db_url = get_db_url(db_config)
        schema = db_config.get("SCHEMA", None)
        pool = NullPool if poolclass == "NullPool" else None

        if schema:
            self._engine = engine or create_engine(
                db_url,
                echo=False,
                connect_args={"options": "-csearch_path={}".format(schema)},
                poolclass=pool,
            )
        else:
            self._engine = engine or create_engine(db_url, echo=False, poolclass=pool)

        self._session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )

    @contextmanager
    def session(self) -> Callable[..., ContextManager[Session]]:
        session: Session = self._session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def get_classifier_ids_by_name(classifier_names: list, psql_connection) -> dict:
    """{classifier_name: {"classifier_id": int, "classifier_version": str}}.

    The `classifier` table's primary key is `classifier_id` alone, so a duplicated
    `classifier_name` is possible and is a deploy error. This is the only place it
    is still visible (the return value is keyed by name), so it raises here —
    design doc §8, assertion 2.
    """
    statement = text(
        "SELECT classifier_id, classifier_name, classifier_version "
        "FROM classifier WHERE classifier_name IN :names"
    ).bindparams(bindparam("names", expanding=True))

    found: dict = {}
    duplicates: set = set()
    with psql_connection.session() as session:
        for row in session.execute(statement, {"names": list(classifier_names)}).mappings():
            name = row["classifier_name"]
            if name in found:
                duplicates.add(name)
            found[name] = {
                "classifier_id": int(row["classifier_id"]),
                "classifier_version": row["classifier_version"],
            }

    if duplicates:
        raise ValueError(
            f"classifier table has more than one row for {sorted(duplicates)}; "
            "cannot resolve a classifier_id unambiguously"
        )
    return found


def get_taxonomy_by_classifier_id(classifier_ids: list, psql_connection) -> dict:
    """{classifier_id: {class_name: class_id}} from the taxonomy table.

    Ordered by "order" per classifier — cosmetic for the dict, kept to match the
    offline reference and the stamp step.
    """
    statement = text(
        "SELECT classifier_id, class_id, class_name FROM taxonomy "
        'WHERE classifier_id IN :classifier_ids ORDER BY classifier_id, "order"'
    ).bindparams(bindparam("classifier_ids", expanding=True))

    maps: dict = {}
    with psql_connection.session() as session:
        rows = session.execute(statement, {"classifier_ids": list(classifier_ids)})
        for row in rows.mappings():
            maps.setdefault(int(row["classifier_id"]), {})[row["class_name"]] = int(
                row["class_id"]
            )
    return maps
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest/test_taxonomy.py -v
```

Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline
git add lc_classification_multisurvey_step
git commit -m "feat(lc_classification_multisurvey_step): read-only classifier and taxonomy queries"
```

---

### Task 5: `resolve_classifiers` and the startup assertions

Implements: spec §8 (four fail-fast startup assertions), §4 (startup order).

**Files:**
- Modify: `lc_classification_multisurvey_step/lc_classification_multisurvey_step/db/db.py`
- Test: `lc_classification_multisurvey_step/tests/unittest/test_taxonomy.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unittest/test_taxonomy.py`:

```python
# --- resolve_classifiers ---------------------------------------------------

FIVE_NAMES = ["b", "b_top", "b_transient", "b_stochastic", "b_periodic"]


def classifier_rows(version="2.1.0", names=None, start_id=41):
    """One classifier row per name, with ids that are deliberately not 5-9."""
    names = names if names is not None else FIVE_NAMES
    return [
        {"classifier_id": start_id + i, "classifier_name": n, "classifier_version": version}
        for i, n in enumerate(names)
    ]


def taxonomy_rows(ids):
    return [
        {"classifier_id": cid, "class_id": 0, "class_name": f"class{cid}"} for cid in ids
    ]


ALL_IDS = [41, 42, 43, 44, 45]


class TestResolveClassifiers:
    def test_returns_ids_by_name_and_taxonomy_by_id(self):
        conn = FakeConnection(classifier_rows(), taxonomy_rows(ALL_IDS))

        ids, taxonomy = db.resolve_classifiers(FIVE_NAMES, "2.1.0", conn)

        assert ids == dict(zip(FIVE_NAMES, ALL_IDS))
        assert taxonomy == {cid: {f"class{cid}": 0} for cid in ALL_IDS}

    def test_taxonomy_is_queried_with_the_resolved_ids(self):
        conn = FakeConnection(classifier_rows(), taxonomy_rows(ALL_IDS))
        db.resolve_classifiers(FIVE_NAMES, "2.1.0", conn)
        _statement, params = conn.session_obj.executed[1]
        assert params == {"classifier_ids": ALL_IDS}

    def test_missing_classifier_name_raises_and_names_it(self):
        conn = FakeConnection(classifier_rows(names=FIVE_NAMES[:4]), taxonomy_rows(ALL_IDS))

        with pytest.raises(ValueError) as exc:
            db.resolve_classifiers(FIVE_NAMES, "2.1.0", conn)

        assert "b_periodic" in str(exc.value)

    def test_empty_taxonomy_for_one_head_raises_and_names_it(self):
        conn = FakeConnection(classifier_rows(), taxonomy_rows(ALL_IDS[:4]))

        with pytest.raises(ValueError) as exc:
            db.resolve_classifiers(FIVE_NAMES, "2.1.0", conn)

        assert "45" in str(exc.value)

    def test_version_mismatch_raises_and_reports_both_versions(self):
        conn = FakeConnection(classifier_rows(version="2.0.0"), taxonomy_rows(ALL_IDS))

        with pytest.raises(ValueError) as exc:
            db.resolve_classifiers(FIVE_NAMES, "2.1.0", conn)

        assert "2.0.0" in str(exc.value)
        assert "2.1.0" in str(exc.value)

    def test_duplicate_name_raises(self):
        rows = classifier_rows()
        rows.append(
            {"classifier_id": 99, "classifier_name": "b", "classifier_version": "2.1.0"}
        )
        conn = FakeConnection(rows, taxonomy_rows(ALL_IDS + [99]))

        with pytest.raises(ValueError, match="more than one row"):
            db.resolve_classifiers(FIVE_NAMES, "2.1.0", conn)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest/test_taxonomy.py -v
```

Expected: the `TestResolveClassifiers` tests fail with `AttributeError: module ... has no attribute 'resolve_classifiers'`.

- [ ] **Step 3: Write minimal implementation**

Append to `lc_classification_multisurvey_step/db/db.py`:

```python
def resolve_classifiers(classifier_names: list, model_version: str, psql_connection):
    """Resolve the head names to ids and fetch their taxonomy, or refuse to start.

    Returns ({classifier_name: classifier_id}, {classifier_id: {class_name: class_id}}).

    Implements the design doc's §8 startup assertions. All four raise: an
    unseeded, partially-seeded, ambiguous or version-skewed classifier/taxonomy is
    a deploy error, and a step that started anyway would silently drop every
    probability it produced or write it against the wrong classifier.

      1. every head name resolved to a row          (here)
      2. no name resolved to more than one row      (get_classifier_ids_by_name)
      3. every resolved id has a non-empty taxonomy (here)
      4. each row's classifier_version == model_version (here)
    """
    found = get_classifier_ids_by_name(classifier_names, psql_connection)

    missing = [name for name in classifier_names if name not in found]
    if missing:
        raise ValueError(
            f"classifier table has no row for {missing}; the BHRF classifier seed "
            "is missing or incomplete in this schema. Refusing to start."
        )

    skewed = {
        name: row["classifier_version"]
        for name, row in found.items()
        if row["classifier_version"] != model_version
    }
    if skewed:
        raise ValueError(
            f"classifier.classifier_version {skewed} does not match MODEL_VERSION "
            f"'{model_version}'; the seeded taxonomy may not match the model's "
            "classes_. Refusing to start."
        )

    classifier_ids = {name: found[name]["classifier_id"] for name in classifier_names}
    taxonomy_maps = get_taxonomy_by_classifier_id(list(classifier_ids.values()), psql_connection)

    unseeded = [cid for cid in classifier_ids.values() if not taxonomy_maps.get(cid)]
    if unseeded:
        raise ValueError(
            f"taxonomy table has no rows for classifier_id(s) {unseeded}; "
            "every probability for those heads would be dropped. Refusing to start."
        )

    log.info("resolved classifier ids %s", classifier_ids)
    return classifier_ids, taxonomy_maps
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest -v
```

Expected: everything green (Task 1: 2, Task 2-3: 24, Task 4: 11, Task 5: 6 → `43 passed`).

- [ ] **Step 5: Commit**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline
git add lc_classification_multisurvey_step
git commit -m "feat(lc_classification_multisurvey_step): fail-fast startup classifier resolution"
```

---

### Task 6: `input_dto.py`

Implements: spec §4 (features-only DTO, `oid` already bigint), §7 (`lastmjd` from detections, no JD subtraction), §8 (messages with no features filtered), §13 (`MIN_DETECTIONS`, non-forced only).

**Files:**
- Create: `lc_classification_multisurvey_step/lc_classification_multisurvey_step/input_dto.py`
- Test: `lc_classification_multisurvey_step/tests/unittest/test_input_dto.py`

`create_input_dto` imports `alerce_classifiers` **inside the function body**, so the module imports and the other three functions test without the submodule.

- [ ] **Step 1: Write the failing test**

Create `tests/unittest/test_input_dto.py`:

```python
"""Message batch -> features frame, lastmjd map, and filtering.

Message shape follows schemas/feature_step/output.avsc: oid is a string carrying
the bigint masterid, detections is an array of records each with an mjd and a
`forced` flag, features is a nullable record (a plain dict here).
"""
import pandas as pd
import pytest

from lc_classification_multisurvey_step import input_dto


def detection(mjd, forced=False):
    return {"mjd": mjd, "forced": forced, "candid": "c", "oid": "1"}


DEFAULT = object()  # sentinel: `features=None` must be able to mean a null record


def message(oid="12345", features=DEFAULT, detections=None):
    return {
        "oid": oid,
        "features": {"feat_a": 1.0, "feat_b": 2.0} if features is DEFAULT else features,
        "detections": detections if detections is not None else [detection(60000.0)],
    }


class TestFilterMessages:
    def test_keeps_messages_with_features(self):
        msgs = [message(oid="1"), message(oid="2")]
        assert len(input_dto.filter_messages(msgs)) == 2

    def test_drops_messages_with_null_features(self):
        msgs = [message(oid="1", features=None), message(oid="2")]
        kept = input_dto.filter_messages(msgs)
        assert [m["oid"] for m in kept] == ["2"]

    def test_drops_messages_with_empty_features(self):
        kept = input_dto.filter_messages([message(oid="1", features={})])
        assert kept == []

    def test_min_detections_unset_keeps_everything(self):
        msgs = [message(detections=[detection(1.0)])]
        assert len(input_dto.filter_messages(msgs, min_detections=None)) == 1

    def test_min_detections_counts_non_forced_only(self):
        msgs = [
            message(
                oid="1",
                detections=[detection(1.0), detection(2.0, forced=True), detection(3.0, forced=True)],
            )
        ]
        assert input_dto.filter_messages(msgs, min_detections=2) == []
        assert len(input_dto.filter_messages(msgs, min_detections=1)) == 1


class TestBuildFeaturesFrame:
    def test_one_row_per_message_indexed_by_int_oid(self):
        msgs = [
            message(oid="12345", features={"a": 1.0, "b": 2.0}),
            message(oid="67890", features={"a": 3.0, "b": 4.0}),
        ]

        frame = input_dto.build_features_frame(msgs)

        assert list(frame.index) == [12345, 67890]
        assert frame.index.name == "oid"
        assert list(frame.columns) == ["a", "b"]
        assert frame.loc[67890, "a"] == 3.0

    def test_oid_is_cast_to_int_not_left_as_string(self):
        frame = input_dto.build_features_frame([message(oid="12345")])
        assert frame.index[0] == 12345
        assert not isinstance(frame.index[0], str)

    def test_empty_batch_gives_an_empty_frame(self):
        frame = input_dto.build_features_frame([])
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == 0
        assert frame.index.name == "oid"

    def test_duplicate_oids_collapse_keeping_the_last_message(self):
        # Two messages for the same object can land in one consume batch. Left
        # alone they would produce two probability rows colliding on
        # (oid, sid, classifier_id, class_id), and the scribe's highest-lastmjd
        # dedup could not break the tie. The stamp step collapses the same way.
        msgs = [
            message(oid="1", features={"a": 1.0}),
            message(oid="1", features={"a": 2.0}),
            message(oid="2", features={"a": 3.0}),
        ]

        frame = input_dto.build_features_frame(msgs)

        assert list(frame.index) == [1, 2]
        assert frame.loc[1, "a"] == 2.0  # last message wins


class TestLastmjdByOid:
    def test_max_mjd_over_detections(self):
        msgs = [message(oid="1", detections=[detection(60000.0), detection(60010.5)])]
        assert input_dto.lastmjd_by_oid(msgs) == {1: 60010.5}

    def test_forced_photometry_counts_toward_lastmjd(self):
        msgs = [message(oid="1", detections=[detection(60000.0), detection(60020.0, forced=True)])]
        assert input_dto.lastmjd_by_oid(msgs) == {1: 60020.0}

    def test_no_jd_offset_is_subtracted(self):
        msgs = [message(oid="1", detections=[detection(60000.0)])]
        assert input_dto.lastmjd_by_oid(msgs)[1] == pytest.approx(60000.0)

    def test_message_without_detections_is_absent(self):
        msgs = [message(oid="1", detections=[]), message(oid="2")]
        assert input_dto.lastmjd_by_oid(msgs) == {2: 60000.0}


class TestCreateInputDto:
    def test_features_only_dto(self):
        pytest.importorskip("alerce_classifiers.base.factories")

        dto = input_dto.create_input_dto([message(oid="1", features={"a": 1.0})])

        assert list(dto.features.index) == [1]
        assert len(dto.detections) == 0
        assert len(dto.non_detections) == 0
        assert len(dto.xmatch) == 0
        assert len(dto.stamps) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest/test_input_dto.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'lc_classification_multisurvey_step.input_dto'`.

- [ ] **Step 3: Write minimal implementation**

Create `lc_classification_multisurvey_step/input_dto.py`:

```python
"""feature_step messages -> features-only InputDTO, plus the lastmjd map.

`SquidwardFeaturesClassifier.can_predict` inspects only `input_dto.features`, and
`predict` calls `mapper.preprocess(input_dto)` which reads only features. So
detections / non-detections / xmatch / stamps are passed empty (design doc §4),
which also drops the legacy step's stale candid schema and its pickled
extra_fields round-trip.

`alerce_classifiers` is imported lazily inside `create_input_dto` so the rest of
this module — and the unit suite — needs no model dependency.
"""
import logging

import pandas as pd

log = logging.getLogger(__name__)


def filter_messages(messages: list, min_detections=None) -> list:
    """Drop messages the classifier cannot or should not consume.

    - no features (`features` is None or empty) -> cannot classify (design §8);
    - fewer than `min_detections` *non-forced* detections -> optional pre-filter,
      counted the way the legacy step counts it (design §13). Unset by default.
    """
    kept = []
    for message in messages:
        if not message.get("features"):
            continue
        if min_detections is not None:
            n_detections = sum(
                1 for d in (message.get("detections") or []) if not d.get("forced", False)
            )
            if n_detections < min_detections:
                continue
        kept.append(message)
    return kept


def build_features_frame(messages: list) -> pd.DataFrame:
    """One row per message, indexed by the bigint oid, columns = feature names.

    The multisurvey feature_step already emits the bigint masterid in `oid` (the
    Avro field is typed string), so this casts with `int()` and calls no idmapper
    — unlike the stamp step, which starts from raw ZTF alerts (design doc §4).

    Duplicate oids within one batch are collapsed, keeping the LAST message for
    that oid. Two messages for the same object can arrive in a single consume
    batch; left alone they would yield two probability rows colliding on
    `(oid, sid, classifier_id, class_id)`, which the scribe's highest-lastmjd
    dedup cannot break because both carry the same lastmjd. This is what upholds
    `build_probability_rows`' unique-oid-index contract.
    """
    if not messages:
        frame = pd.DataFrame()
        frame.index.name = "oid"
        return frame

    frame = pd.DataFrame(
        [message["features"] for message in messages],
        index=[int(message["oid"]) for message in messages],
    )
    frame.index.name = "oid"
    return frame[~frame.index.duplicated(keep="last")]


def lastmjd_by_oid(messages: list) -> dict:
    """{oid: max detection mjd}. Already MJD — do NOT subtract 2400000.5.

    The `detections` array carries forced photometry too (each entry has a
    `forced` flag), so this is the max over detections and forced together,
    matching offline `classify._lc_lastmjd`.
    """
    lastmjd = {}
    for message in messages:
        mjds = [
            float(d["mjd"])
            for d in (message.get("detections") or [])
            if d.get("mjd") is not None
        ]
        if not mjds:
            log.warning("oid=%s has no detection mjd; it will produce no rows", message["oid"])
            continue
        lastmjd[int(message["oid"])] = max(mjds)
    return lastmjd


def create_input_dto(messages: list):
    """Features-only InputDTO for the batch."""
    from alerce_classifiers.base.factories import input_dto_factory

    empty = pd.DataFrame()
    return input_dto_factory(empty, empty, build_features_frame(messages), empty, empty)
```

- [ ] **Step 4: Run test to verify it passes**

Without the submodule on the path, the DTO test skips; with it, it runs. Do both:

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest/test_input_dto.py -v
PYTHONPATH=/home/fandrades/desktop/pipeline_features/pipeline/alerce_classifiers \
  /home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest/test_input_dto.py -v
```

Expected: first run `13 passed, 1 skipped`; second run `14 passed`.

- [ ] **Step 5: Commit**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline
git add lc_classification_multisurvey_step
git commit -m "feat(lc_classification_multisurvey_step): features-only input DTO and lastmjd map"
```

---

### Task 7: `output_parser.py` (placeholder)

Implements: spec §9 (placeholder downstream output).

**Files:**
- Create: `lc_classification_multisurvey_step/lc_classification_multisurvey_step/output_parser.py`
- Test: `lc_classification_multisurvey_step/tests/unittest/test_output_parser.py`

This shape is explicitly **not** a contract (spec §9): no `.avsc` is added, and nothing downstream should consume it until the schema is designed. It is tested only so it cannot crash the step.

- [ ] **Step 1: Write the failing test**

Create `tests/unittest/test_output_parser.py`:

```python
"""The downstream payload is a PLACEHOLDER (design doc §9). These tests pin only
that it is well-formed and cannot throw — not that the shape is a contract."""
from types import SimpleNamespace

import pandas as pd

from lc_classification_multisurvey_step.output_parser import MultisurveyOutputParser


def frame(index, data):
    df = pd.DataFrame(data, index=index)
    df.index.name = "oid"
    return df


def make_dto(flat, top=None):
    return SimpleNamespace(
        probabilities=flat,
        hierarchical={"top": top, "children": {}},
    )


class TestMultisurveyOutputParser:
    def test_one_message_per_oid_with_top_class_per_head(self):
        dto = make_dto(
            flat=frame([1, 2], {"SNIa": [0.9, 0.1], "AGN": [0.1, 0.9]}),
            top=frame([1, 2], {"Transient": [0.8, 0.2], "Stochastic": [0.2, 0.8]}),
        )

        out = MultisurveyOutputParser().parse(dto, base_name="base", version="2.1.0").value

        assert [m["oid"] for m in out] == [1, 2]
        first = out[0]
        assert first["classifier_name"] == "base"
        assert first["classifier_version"] == "2.1.0"
        assert first["top_class"]["base"] == {"class_name": "SNIa", "probability": 0.9}
        assert first["top_class"]["base_top"]["class_name"] == "Transient"
        assert out[1]["top_class"]["base"]["class_name"] == "AGN"

    def test_missing_heads_are_absent_not_null(self):
        dto = make_dto(flat=frame([1], {"SNIa": [1.0]}), top=None)
        out = MultisurveyOutputParser().parse(dto, base_name="base", version="2.1.0").value
        assert list(out[0]["top_class"]) == ["base"]

    def test_empty_output_dto_produces_no_messages(self):
        dto = make_dto(flat=pd.DataFrame())
        assert MultisurveyOutputParser().parse(dto, base_name="base", version="2.1.0").value == []

    def test_none_output_dto_produces_no_messages(self):
        assert MultisurveyOutputParser().parse(None, base_name="base", version="2.1.0").value == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest/test_output_parser.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'lc_classification_multisurvey_step.output_parser'`.

- [ ] **Step 3: Write minimal implementation**

Create `lc_classification_multisurvey_step/output_parser.py`:

```python
"""PLACEHOLDER downstream producer payload.

Decision 5 of the design doc is "new multisurvey output schema", but its shape is
deferred — see §9. This emits a minimal per-oid message (oid, classifier name and
version, and the top-ranked class per head) so the step's produce stage is wired
end to end. It is NOT a contract: no schemas/lc_classification_multisurvey_step/
avsc exists, and nothing downstream should be pointed at this topic until the
schema is designed. Deferring is safe because the scribe is the real output path
(decision 3).

Duck-typed over the OutputDTO like probabilities.py, so it needs no
alerce_classifiers import.
"""
import logging
from dataclasses import dataclass
from typing import Generic, TypeVar

from .probabilities import DEFAULT_CLASSIFIER_NAME, iter_head_frames

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class KafkaOutput(Generic[T]):
    value: T


class MultisurveyOutputParser:
    """OutputDTO -> placeholder downstream messages."""

    def parse(
        self, model_output, base_name: str = DEFAULT_CLASSIFIER_NAME, version: str = "", **kwargs
    ) -> KafkaOutput:
        """One message per oid in `model_output.probabilities`.

        Caller contract, as for `build_probability_rows`: each head's frame must
        have a unique oid index. Duplicate oids are not collapsed here;
        de-duplication happens upstream when the features frame is built.

        A head is absent from an oid's `top_class` (never present as null) when
        the head is missing, has no classes, does not cover that oid, or scored
        it entirely NaN. Per design §8 these drop the affected (oid, head) entry
        instead of killing the batch.
        """
        if model_output is None or model_output.probabilities is None:
            return KafkaOutput([])
        if len(model_output.probabilities) == 0:
            return KafkaOutput([])

        # Rank each head once for the whole frame. The per-oid form (a .loc plus
        # an idxmax per oid per head) costs ~24x more on a 1000-object batch
        # (344 ms vs 15 ms, same output).
        # Logged once per head, never per oid: a per-oid line is a thousand
        # lines a batch.
        heads = []
        for name, frame in iter_head_frames(model_output, base_name):
            # A head that scored nobody is routine (no oid took that branch),
            # so it is dropped without a warning.
            if frame is None or frame.shape[0] == 0:
                continue
            # Rows but no classes is not routine. The dropna below would empty
            # this frame anyway; the explicit guard is here to name the case and
            # say so, rather than let a broken head vanish quietly.
            if frame.shape[1] == 0:
                log.warning(
                    "head '%s': frame has no classes; dropping the head for all %d oids "
                    "in the batch",
                    name,
                    frame.shape[0],
                )
                continue
            # An oid scored entirely NaN has no argmax: pandas 2 returns NaN (an
            # opaque KeyError downstream) and pandas 3 raises for the whole head.
            # Dropping those rows leaves just those oids uncovered by this head.
            # how="all", not "any": an oid with a NaN in only some classes still
            # has a valid winner and must keep it.
            scored = frame.dropna(how="all")
            unscored = frame.shape[0] - scored.shape[0]
            if unscored:
                log.warning(
                    "head '%s': %d of %d oids scored entirely NaN; dropping the head "
                    "for those oids",
                    name,
                    unscored,
                    frame.shape[0],
                )
            frame = scored
            if frame.shape[0] == 0:
                continue
            # Plain dicts, not Series: the per-oid lookup below is then a hash
            # rather than a pandas label lookup, which is ~4x cheaper again.
            heads.append(
                (name, frame.idxmax(axis=1).to_dict(), frame.max(axis=1).to_dict())
            )

        messages = []
        for oid in model_output.probabilities.index:
            top_class = {}
            for name, class_names, probabilities in heads:
                if oid not in class_names:
                    continue
                top_class[name] = {
                    "class_name": str(class_names[oid]),
                    "probability": float(probabilities[oid]),
                }
            messages.append(
                {
                    "oid": int(oid),
                    "classifier_name": base_name,
                    "classifier_version": version,
                    "top_class": top_class,
                }
            )
        return KafkaOutput(messages)
```

> **Note on this block.** The version above is the as-built code, updated after
> review. The originally-planned form ranked each head per oid (`frame.loc[oid]`
> then `series.idxmax()`), which measured **344 ms** on a 1000-object batch
> against **15 ms** for this one — the same per-row-loop mistake Task 3 had to
> undo in `probabilities.py`. It also crashed the batch on a head frame with rows
> but zero columns, and on an all-NaN row. Prefer the head-level form when
> writing anything similar: rank once per frame, then look up per oid.

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest -v
```

Expected: everything green — `73 passed, 1 skipped`, the DTO test being the one that skips without the submodule on the path.

> **On the counts below.** The per-task test counts written into this plan were
> estimates made before implementation. Review rounds added tests — regression
> tests for real defects found in Tasks 3, 5, and 6 — so the suite is larger than
> originally projected. `73 passed, 1 skipped` after Task 7 is the as-built
> number, verified by running it. If your count is higher, that is expected when
> review adds a test; reconcile it explicitly rather than deleting tests to hit
> a number in this document.

- [ ] **Step 5: Commit**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline
git add lc_classification_multisurvey_step
git commit -m "feat(lc_classification_multisurvey_step): placeholder downstream output parser"
```

---

### Task 8: `settings.py`, `models_settings.py`

Implements: spec §3 (`models_settings` one entry), §10 (configuration).

**Files:**
- Create: `lc_classification_multisurvey_step/models_settings.py`
- Create: `lc_classification_multisurvey_step/settings.py`

No unit test — these are env plumbing, verified by importing them under a controlled environment. Task 9 depends on the exact key names below.

> **`credentials.py` is deliberately NOT created.** An earlier draft of this plan
> copied it from `correction_multisurvey_step` "for parity". It would be dead
> code: the sibling's `run_step.py` imports `get_credentials`, but this step's
> (Task 10) does not, and `settings.py` reads `PSQL_CONFIG` straight from env
> vars as spec §10 specifies. The file is also almost entirely a MongoDB config
> parser, and this step touches no MongoDB. Copying it would imply a
> secret-manager path that does not exist here. If a secret-manager path is
> wanted later, add it deliberately with the step that uses it.

- [ ] **Step 1: Write `models_settings.py`**

```python
import os


def squidward_params(model_class: str):
    """The single BHRF entry. Mirrors the stamp step's one-entry configurator so
    the model class, mapper class and model path stay env-driven (design §3)."""
    return {
        "CLASS": model_class,
        "CLASS_MAPPER": os.getenv("CLASS_MAPPER"),
        "PARAMS": {"model_path": os.getenv("MODEL_PATH")},
        "NAME": model_class.split(".")[-1],
        "VERSION": os.getenv("MODEL_VERSION", "2.1.0"),
        "CLASSIFIER_NAME": os.getenv("CLASSIFIER_NAME", "lc_classifier_BHRF_forced_phot"),
        "SID": int(os.getenv("SID", 0)),
        "MIN_DETECTIONS": (
            int(os.environ["MIN_DETECTIONS"]) if os.getenv("MIN_DETECTIONS") else None
        ),
    }


def configurator(model_class: str):
    if model_class.endswith("SquidwardFeaturesClassifier"):
        return squidward_params(model_class)

    raise Exception(f"Model class not supported by this step: {model_class}")
```

- [ ] **Step 2: Write `settings.py`**

```python
##################################################
#   LC Classification Multisurvey Settings File
##################################################
import os
import pathlib

from models_settings import configurator


def model_config_factory():
    return configurator(os.environ["MODEL_CLASS"])


def config():
    CONSUMER_CONFIG = {
        "CLASS": os.getenv("CONSUMER_CLASS", "apf.consumers.KafkaConsumer"),
        "TOPICS": os.environ["CONSUMER_TOPICS"].strip().split(","),
        "PARAMS": {
            "bootstrap.servers": os.environ["CONSUMER_SERVER"],
            "group.id": os.environ["CONSUMER_GROUP_ID"],
            "auto.offset.reset": "beginning",
            "enable.partition.eof": bool(os.getenv("ENABLE_PARTITION_EOF", None)),
        },
        "consume.timeout": int(os.getenv("CONSUME_TIMEOUT", 10)),
        "consume.messages": int(os.getenv("CONSUME_MESSAGES", 100)),
    }

    scribe_schema_path = str(
        pathlib.Path(
            pathlib.Path(__file__).parent.parent, "schemas/scribe_step", "scribe.avsc"
        )
    )
    SCRIBE_PRODUCER_CONFIG = {
        "CLASS": os.getenv("SCRIBE_PRODUCER_CLASS", "apf.producers.KafkaProducer"),
        "PARAMS": {"bootstrap.servers": os.environ["SCRIBE_SERVER"]},
        "TOPIC": os.environ["SCRIBE_TOPIC"],
        "SCHEMA_PATH": os.getenv("SCRIBE_SCHEMA_PATH", scribe_schema_path),
    }

    # PLACEHOLDER downstream output (design §9): no schema is defined yet, so the
    # producer is only configured when PRODUCER_SERVER is set. Without it apf
    # falls back to its DefaultProducer and the step produces nothing downstream.
    PRODUCER_CONFIG = {}
    if os.getenv("PRODUCER_SERVER"):
        PRODUCER_CONFIG = {
            "CLASS": os.getenv("PRODUCER_CLASS", "apf.producers.kafka.KafkaProducer"),
            "PARAMS": {"bootstrap.servers": os.environ["PRODUCER_SERVER"]},
            "TOPIC": os.environ["PRODUCER_TOPIC"],
        }

    METRICS_CONFIG = {}
    if os.getenv("METRICS_HOST"):
        metrics_schema_path = str(
            pathlib.Path(
                pathlib.Path(__file__).parent.parent,
                "schemas/lc_classification_step",
                "metrics.json",
            )
        )
        METRICS_CONFIG = {
            "CLASS": "apf.metrics.KafkaMetricsProducer",
            "PARAMS": {
                "PARAMS": {"bootstrap.servers": os.environ["METRICS_HOST"]},
                "TOPIC": os.environ["METRICS_TOPIC"],
                "SCHEMA_PATH": os.getenv("METRICS_SCHEMA_PATH", metrics_schema_path),
            },
        }

    PSQL_CONFIG = {
        "HOST": os.environ["PSQL_HOST"],
        "USER": os.environ["PSQL_USER"],
        "PASSWORD": os.environ["PSQL_PASSWORD"],
        "PORT": int(os.getenv("PSQL_PORT", 5432)),
        "DB_NAME": os.environ["PSQL_DATABASE"],
        "SCHEMA": os.getenv("PSQL_SCHEMA", "multisurvey_ztf"),
    }

    if os.getenv("CONSUMER_KAFKA_USERNAME") and os.getenv("CONSUMER_KAFKA_PASSWORD"):
        CONSUMER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
        CONSUMER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        CONSUMER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("CONSUMER_KAFKA_USERNAME")
        CONSUMER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("CONSUMER_KAFKA_PASSWORD")
    if os.getenv("SCRIBE_KAFKA_USERNAME") and os.getenv("SCRIBE_KAFKA_PASSWORD"):
        SCRIBE_PRODUCER_CONFIG["PARAMS"]["security.protocol"] = "SASL_SSL"
        SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.mechanism"] = "SCRAM-SHA-512"
        SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.username"] = os.getenv("SCRIBE_KAFKA_USERNAME")
        SCRIBE_PRODUCER_CONFIG["PARAMS"]["sasl.password"] = os.getenv("SCRIBE_KAFKA_PASSWORD")

    return {
        "CONSUMER_CONFIG": CONSUMER_CONFIG,
        "PRODUCER_CONFIG": PRODUCER_CONFIG,
        "SCRIBE_PRODUCER_CONFIG": SCRIBE_PRODUCER_CONFIG,
        "METRICS_CONFIG": METRICS_CONFIG,
        "PSQL_CONFIG": PSQL_CONFIG,
        "MODEL_CONFIG": model_config_factory(),
        "FEATURE_FLAGS": {
            "PROMETHEUS": bool(os.getenv("USE_PROMETHEUS", False)),
        },
        "LOGGING_DEBUG": bool(os.getenv("LOGGING_DEBUG", False)),
    }
```

Note `PSQL_CONFIG` uses the key `USER` (not `USERNAME`): `db.get_db_url` reads `config['USER']`, matching `stamp_classifier_2025_multisurvey_step`. `correction_multisurvey_step` uses `USERNAME` with its own connection class — do not cross the two.

- [ ] **Step 3: Verify settings builds under a controlled environment**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline/lc_classification_multisurvey_step
env -i PATH=$PATH \
  CONSUMER_TOPICS=features CONSUMER_SERVER=localhost:9092 CONSUMER_GROUP_ID=g \
  SCRIBE_SERVER=localhost:9092 SCRIBE_TOPIC=scribe \
  PSQL_HOST=h PSQL_USER=u PSQL_PASSWORD=p PSQL_DATABASE=d \
  MODEL_CLASS=alerce_classifiers.squidward.model.SquidwardFeaturesClassifier \
  CLASS_MAPPER=alerce_classifiers.squidward.mapper.SquidwardMapper \
  MODEL_PATH=s3://fake/model.pkl \
  /home/fandrades/miniconda3/envs/feature_step/bin/python -c "
import settings
c = settings.config()
assert c['MODEL_CONFIG']['VERSION'] == '2.1.0', c['MODEL_CONFIG']
assert c['MODEL_CONFIG']['CLASSIFIER_NAME'] == 'lc_classifier_BHRF_forced_phot'
assert c['MODEL_CONFIG']['MIN_DETECTIONS'] is None
assert c['MODEL_CONFIG']['SID'] == 0
assert c['PSQL_CONFIG']['SCHEMA'] == 'multisurvey_ztf'
assert c['PRODUCER_CONFIG'] == {}
print('OK', c['MODEL_CONFIG']['NAME'])
"
```

Expected: `OK SquidwardFeaturesClassifier`.

- [ ] **Step 4: Verify an unsupported model class is rejected**

```bash
env -i PATH=$PATH MODEL_CLASS=some.other.Model \
  /home/fandrades/miniconda3/envs/feature_step/bin/python -c "
from models_settings import configurator
try:
    configurator('some.other.Model')
except Exception as e:
    print('OK rejected:', e); raise SystemExit(0)
raise SystemExit('should have raised')
"
```

Expected: `OK rejected: Model class not supported by this step: some.other.Model`.

- [ ] **Step 5: Commit**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline
git add lc_classification_multisurvey_step
git commit -m "feat(lc_classification_multisurvey_step): settings and model configuration"
```

---

### Task 9: `step.py`

Implements: spec §4 (data flow, startup), §7 (scribe command envelope), §8 (`can_predict` false → produce nothing).

**Files:**
- Create: `lc_classification_multisurvey_step/lc_classification_multisurvey_step/step.py`

This is the only module importing apf and alerce_classifiers. It holds wiring, not logic — everything worth asserting already has tests in Tasks 3, 5, 6, 7.

- [ ] **Step 1: Write the implementation**

Create `lc_classification_multisurvey_step/step.py`:

```python
"""Multisurvey LC classification step (ZTF BHRF / Squidward).

Consumes the multisurvey feature_step output topic, classifies the batch, and
produces one `update-probability` command per probability row to the
scribe_multisurvey topic. The step writes nothing to the database — the scribe
owns the upsert (design doc §2, decision 3).
"""
import json
import logging
import traceback
from typing import List, Tuple

import numexpr
import pandas as pd
from apf.consumers import KafkaConsumer
from apf.core import get_class
from apf.core.step import GenericStep
from alerce_classifiers.base.dto import OutputDTO

from .db.db import PSQLConnection, resolve_classifiers
from .input_dto import create_input_dto, filter_messages, lastmjd_by_oid
from .output_parser import MultisurveyOutputParser
from .probabilities import build_probability_rows, head_names


class LateClassifierMultisurvey(GenericStep):
    """BHRF classification over the multisurvey feature stream."""

    def __init__(self, config={}, level=logging.INFO, **step_args):
        super().__init__(config=config, level=level, **step_args)
        numexpr.utils.set_num_threads(1)

        model_config = config["MODEL_CONFIG"]
        self.classifier_name = model_config["CLASSIFIER_NAME"]
        self.model_version = model_config["VERSION"]
        self.sid = int(model_config.get("SID", 0))
        self.min_detections = model_config.get("MIN_DETECTIONS")

        # Startup, in order: names -> ids, then ids -> taxonomy. Both are
        # read-only and cached; the connection is not used again. Any of the four
        # §8 assertions failing raises here and the step refuses to start.
        #
        # NullPool because this connection serves exactly two queries and is then
        # idle for the life of the consumer. With the default QueuePool the
        # startup checkout is returned to the pool rather than closed, holding an
        # idle Postgres connection per replica against max_connections for
        # nothing. correction_multisurvey_step/step.py does the same.
        self.db = PSQLConnection(config["PSQL_CONFIG"], poolclass="NullPool")
        self.classifier_ids, self.taxonomy_maps = resolve_classifiers(
            head_names(self.classifier_name), self.model_version, self.db
        )

        self.mapper = get_class(model_config["CLASS_MAPPER"])()
        self.model = get_class(model_config["CLASS"])(
            **{"mapper": self.mapper, **model_config["PARAMS"]}
        )

        scribe_config = config["SCRIBE_PRODUCER_CONFIG"]
        self.scribe_producer = get_class(scribe_config["CLASS"])(scribe_config)

        self.step_parser = MultisurveyOutputParser()

    @staticmethod
    def _empty_output() -> OutputDTO:
        return OutputDTO(pd.DataFrame(), {"top": pd.DataFrame(), "children": {}})

    def execute(self, messages: List[dict]) -> Tuple[OutputDTO, dict]:
        kept = filter_messages(messages, self.min_detections)
        self.logger.info(f"Classifying {len(kept)}/{len(messages)} messages")
        if not kept:
            return self._empty_output(), {}

        dto = create_input_dto(kept)
        can_predict, reason = self.model.can_predict(dto)
        if not can_predict:
            self.logger.warning(f"Model cannot predict this batch: {reason}")
            return self._empty_output(), {}

        try:
            output_dto = self.model.predict(dto)
        except Exception as e:
            self.logger.error(f"Prediction failed for this batch: {e}")
            self.logger.error(traceback.format_exc())
            return self._empty_output(), {}

        return output_dto, lastmjd_by_oid(kept)

    def post_execute(self, result: Tuple[OutputDTO, dict]) -> Tuple[OutputDTO, dict]:
        output_dto, lastmjd_map = result
        rows = build_probability_rows(
            output_dto,
            lastmjd_map,
            self.classifier_ids,
            self.taxonomy_maps,
            base_name=self.classifier_name,
            version=self.model_version,
            sid=self.sid,
        )
        self.produce_scribe(rows)
        return result

    def produce_scribe(self, rows: List[dict]) -> None:
        """One `update-probability` command per row, keyed by oid.

        Envelope matches stamp_classifier_2025_multisurvey_step and is accepted by
        scribe_multisurvey's `decode.command_factory` (design doc §7).
        """
        if not rows:
            return

        last_index = len(rows) - 1
        for index, row in enumerate(rows):
            command = {"step": "update-probability", "survey": "ztf", "payload": row}
            self.scribe_producer.produce(
                {"payload": json.dumps(command)},
                key=str(row["oid"]).encode("utf-8"),
                on_delivery=None,
            )
            if index == last_index:
                self.scribe_producer.producer.flush()

        self.logger.info(f"Produced {len(rows)} probability rows to the scribe")

    def pre_produce(self, result: Tuple[OutputDTO, dict]):
        # PLACEHOLDER downstream payload — design doc §9.
        return self.step_parser.parse(
            result[0], base_name=self.classifier_name, version=self.model_version
        ).value

    def tear_down(self):
        if isinstance(self.consumer, KafkaConsumer):
            self.consumer.teardown()
        else:
            self.consumer.__del__()
```

- [ ] **Step 2: Verify it imports**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline/lc_classification_multisurvey_step
PYTHONPATH=/home/fandrades/desktop/pipeline_features/pipeline/alerce_classifiers \
  /home/fandrades/miniconda3/envs/feature_step/bin/python -c "
from lc_classification_multisurvey_step.step import LateClassifierMultisurvey
print('OK', LateClassifierMultisurvey.__name__)
"
```

Expected: `OK LateClassifierMultisurvey`.

> **`ModuleNotFoundError: No module named 'numexpr'` is expected on this machine
> and is NOT a code defect.** An earlier draft of this note blamed the
> interpreter; that was wrong. `numexpr` is absent from all eleven conda envs on
> this box, and there is no poetry `.venv` for the step. The dependency is
> correctly declared (`pyproject.toml`: `numexpr = "^2.8.8"`) and all four
> sibling classifier steps do the identical `import numexpr` +
> `numexpr.utils.set_num_threads(1)`, so the container build installs it and
> production is unaffected.
>
> Do **not** remove the import to make this check pass. To verify the rest of
> the import chain locally, either `pip install numexpr` into the env, or put a
> throwaway `numexpr` stub on `PYTHONPATH` (outside the project tree) — with a
> stub the command prints `OK LateClassifierMultisurvey`, confirming every other
> import resolves.

- [ ] **Step 3: Verify the unit suite still passes untouched**

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/unittest -v
```

Expected: `78 passed, 1 skipped` — unchanged from Task 7. If the count changed, `step.py` leaked an import into a pure module.

- [ ] **Step 4: Commit**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline
git add lc_classification_multisurvey_step
git commit -m "feat(lc_classification_multisurvey_step): step wiring"
```

---

### Task 10: `scripts/run_step.py` and `Dockerfile`

Implements: spec §3 (module layout).

**Files:**
- Create: `lc_classification_multisurvey_step/scripts/run_step.py`
- Create: `lc_classification_multisurvey_step/Dockerfile`

- [ ] **Step 1: Write `scripts/run_step.py`**

Follows `correction_multisurvey_step/scripts/run_step.py`'s yaml-or-settings pattern rather than the stamp step's yaml-only one, so the step is runnable locally from env vars.

```python
import logging
import os
import sys

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
sys.path.append(PACKAGE_PATH)

from apf.core.settings import config_from_yaml_file  # noqa: E402


def set_logger(settings):
    level = logging.DEBUG if settings.get("LOGGING_DEBUG") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return level


def step_creator():
    if os.getenv("CONFIG_FROM_YAML", False):
        settings = config_from_yaml_file("/config/config.yaml")
    else:
        from settings import config

        settings = config()

    level = set_logger(settings)

    if settings.get("FEATURE_FLAGS", {}).get("PROMETHEUS"):
        from prometheus_client import start_http_server

        start_http_server(8000)

    from lc_classification_multisurvey_step.step import LateClassifierMultisurvey

    return LateClassifierMultisurvey(config=settings, level=level)


def step():
    step_creator().start()


if __name__ == "__main__":
    step()
```

- [ ] **Step 2: Write the `Dockerfile`**

Adapted from `stamp_classifier_2025_multisurvey_step/Dockerfile`, dropping `libs/idmapper` (this step never maps oids — see spec §4, the feature_step already emits the bigint), dropping `schemas/ztf` and the stamp schema dir, and keeping `schemas/scribe_step` plus `schemas/lc_classification_step` (the latter only for the `metrics.json` that `settings.py` points at). Note the build context is the **repo root**, not the step directory.

Create `lc_classification_multisurvey_step/Dockerfile`:

```dockerfile
FROM python:3.10-slim as python-base
LABEL org.opencontainers.image.authors="ALeRCE"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1


FROM python-base as builder
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install poetry
COPY lc_classification_multisurvey_step/pyproject.toml /app/
COPY libs/apf /libs/apf
COPY libs/db-plugins-multisurvey /libs/db-plugins-multisurvey
COPY alerce_classifiers /alerce_classifiers
COPY schemas/scribe_step /schemas/scribe_step
COPY schemas/lc_classification_step /schemas/lc_classification_step
WORKDIR /app
RUN poetry install --no-root


FROM python:3.10-slim as production
RUN pip install poetry
COPY --from=builder /app /app
COPY --from=builder /libs/apf /libs/apf
COPY --from=builder /libs/db-plugins-multisurvey /libs/db-plugins-multisurvey
COPY --from=builder /alerce_classifiers /alerce_classifiers
COPY --from=builder /schemas /schemas
COPY lc_classification_multisurvey_step/scripts /app/scripts
COPY lc_classification_multisurvey_step/README.md /app/README.md
COPY lc_classification_multisurvey_step/settings.py /app/settings.py
COPY lc_classification_multisurvey_step/models_settings.py /app/models_settings.py
COPY lc_classification_multisurvey_step/lc_classification_multisurvey_step /app/lc_classification_multisurvey_step

WORKDIR /app/
RUN poetry install --only-root
CMD ["poetry", "run", "python", "scripts/run_step.py"]
```

Do **not** build the image as part of this task — it needs network access and a poetry resolve of the `ztf` extra. Verify only that the paths it references exist:

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline
for p in libs/apf libs/db-plugins-multisurvey alerce_classifiers/alerce_classifiers \
         schemas/scribe_step schemas/lc_classification_step/metrics.json \
         lc_classification_multisurvey_step/scripts; do
  [ -e "$p" ] && echo "ok   $p" || echo "MISS $p"
done
```

Expected: every line `ok`. A `MISS alerce_classifiers/alerce_classifiers` means the submodule is not initialised — run `git submodule update --init alerce_classifiers`.

- [ ] **Step 3: Verify the entrypoint resolves without starting Kafka**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline/lc_classification_multisurvey_step
PYTHONPATH=/home/fandrades/desktop/pipeline_features/pipeline/alerce_classifiers \
  /home/fandrades/miniconda3/envs/feature_step/bin/python -c "
import sys; sys.argv=['x']
sys.path.insert(0, 'scripts')
import run_step
assert callable(run_step.step)
print('OK entrypoint importable')
"
```

Expected: `OK entrypoint importable`. It must not connect to anything — `step_creator()` is not called.

- [ ] **Step 4: Commit**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline
git add lc_classification_multisurvey_step
git commit -m "feat(lc_classification_multisurvey_step): entrypoint and dockerfile"
```

---

### Task 11: Opt-in offline-equivalence test

Implements: spec §11 (equivalence test against the offline reference).

**Files:**
- Create: `lc_classification_multisurvey_step/tests/integration/test_offline_equivalence.py`

This is the test that actually protects the port; the unit tests only check the port's internal consistency. It is opt-in because it needs the `alerce_classifiers` submodule, `MODEL_PATH`, and the offline checkout at `~/desktop/pipeline`.

- [ ] **Step 1: Write the test**

```python
"""Equivalence against the offline reference implementation.

The step's batched row builder and offline `probability_writer.build_probability_rows`
must agree. Offline is strictly per-oid and pins CLASSIFIER_IDS = [5..9]; the step
melts by oid and resolves ids from the DB. So this feeds one OutputDTO through
both and compares row sets modulo ordering.

Opt-in: needs RUN_EQUIVALENCE_TEST=1, the alerce_classifiers submodule importable,
and the offline checkout on the path. Not part of the default unit run.
"""
import importlib.util
import os
import sys

import pytest

OFFLINE_ROOT = os.path.expanduser("~/desktop/pipeline")

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_EQUIVALENCE_TEST"),
    reason="opt-in: set RUN_EQUIVALENCE_TEST=1 (needs the offline checkout)",
)


@pytest.fixture(scope="module")
def offline_writer():
    if not os.path.isdir(OFFLINE_ROOT):
        pytest.skip(f"offline checkout not found at {OFFLINE_ROOT}")
    if OFFLINE_ROOT not in sys.path:
        sys.path.insert(0, OFFLINE_ROOT)
    if importlib.util.find_spec("features.offline.probability_writer") is None:
        pytest.skip("features.offline.probability_writer not importable")
    from features.offline import probability_writer

    return probability_writer


@pytest.fixture
def dto():
    """A single-oid BHRF-shaped OutputDTO built from the real class names."""
    import pandas as pd
    from types import SimpleNamespace

    def frame(data):
        df = pd.DataFrame(data, index=[123456789])
        df.index.name = "oid"
        return df

    return SimpleNamespace(
        probabilities=frame({"SNIa": [0.6], "AGN": [0.3], "LPV": [0.1]}),
        hierarchical={
            "top": frame({"Transient": [0.7], "Stochastic": [0.2], "Periodic": [0.1]}),
            "children": {
                "Transient": frame({"SNIa": [0.8], "SLSN": [0.2]}),
                "Stochastic": frame({"AGN": [0.9], "QSO": [0.1]}),
                "Periodic": frame({"LPV": [0.5], "EA": [0.5]}),
            },
        },
    )


@pytest.fixture
def taxonomy_maps():
    """Mirrors the offline classifier_taxonomy_lut ids 5-9 for the classes above."""
    return {
        5: {"SNIa": 0, "AGN": 1, "LPV": 2},
        6: {"Transient": 0, "Stochastic": 1, "Periodic": 2},
        7: {"SNIa": 0, "SLSN": 1},
        8: {"AGN": 0, "QSO": 1},
        9: {"LPV": 0, "EA": 1},
    }


def test_row_sets_match_offline(offline_writer, dto, taxonomy_maps):
    from lc_classification_multisurvey_step.probabilities import (
        build_probability_rows,
        head_names,
    )

    oid, lastmjd = 123456789, 60123.5
    # Offline pins ids 5-9; the step resolves them. Bind the step's head names to
    # those same ids so the comparison isolates the row-building logic.
    classifier_ids = dict(zip(head_names(), [5, 6, 7, 8, 9]))

    offline_rows = offline_writer.build_probability_rows(
        dto, oid, lastmjd, taxonomy_maps, version="2.1.0", sid=0
    )
    step_rows = build_probability_rows(
        dto, {oid: lastmjd}, classifier_ids, taxonomy_maps, version="2.1.0", sid=0
    )

    def key(row):
        return tuple(sorted(row.items()))

    assert sorted(map(key, step_rows)) == sorted(map(key, offline_rows))
    assert len(step_rows) == 12  # 3 + 3 + 2 + 2 + 2
```

- [ ] **Step 2: Run it both ways**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline/lc_classification_multisurvey_step
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/integration -v
RUN_EQUIVALENCE_TEST=1 /home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests/integration -v
```

Expected: first `1 skipped`; second either `1 passed` or a skip naming the missing offline checkout. **If it fails on row content, that is a real port defect — report it, do not adjust the test to match.**

- [ ] **Step 3: Run the whole suite**

```bash
/home/fandrades/miniconda3/envs/feature_step/bin/python -m pytest tests -v
```

Expected: `73 passed, 2 skipped`.

- [ ] **Step 4: Commit**

```bash
cd /home/fandrades/desktop/pipeline_features/pipeline
git add lc_classification_multisurvey_step
git commit -m "test(lc_classification_multisurvey_step): opt-in offline equivalence test"
```

---

## Out of scope (spec §12)

Do not touch these while executing this plan; if one seems necessary, stop and report:

- `scribe_multisurvey` — its `ProbabilityCommand` already accepts this payload.
- The legacy `lc_classification_step`.
- A downstream output Avro schema (§9 is a deliberate placeholder).
- Seeding `classifier` / `taxonomy` rows, or back-porting them to db-plugins.
- LSST / Rubin models.
- `charts/lc_classification_multisurvey_step/` — deployment is a follow-up.

## Known deviations from the spec, decided while planning

Raise these in review rather than rediscovering them:

1. **`get_classifier_ids_by_name` returns `dict[str, dict]`**, not `dict[str, int]` — the row's `classifier_version` is needed for the §8 version-skew assertion. The spec was updated to match.
2. **The duplicate-name assertion lives in the reader, not `resolve_classifiers`** — a duplicate is only visible before the rows collapse into a name-keyed dict. The other four §8 assertions are in `resolve_classifiers`.
3. **`probabilities.py` also drops a head with no resolved id / no taxonomy map per batch**, logging rather than raising. Startup already guarantees both exist, so this is defence in depth for a caller that skipped `resolve_classifiers`.
4. **No secret-manager path is wired into `settings.py`** — `PSQL_CONFIG` is read straight from env vars, and the spec's §10 lists only `PSQL_*` env vars, so this matches it. `credentials.py` is not created at all (see Task 8): nothing in this step would import it.
5. **`PRODUCER_CONFIG` is empty unless `PRODUCER_SERVER` is set**, letting apf fall back to `DefaultProducer`. The spec says the producer is "configured but the shape is not a contract"; configuring a Kafka producer with no schema would fail at startup, so it is opt-in instead.
