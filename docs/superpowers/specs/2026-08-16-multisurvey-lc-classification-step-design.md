# Multisurvey LC classification step — design

**Branch:** `feat/multisurvey-lc-classification-step` (off `multisurvey`)
**Date:** 2026-08-16
**Status:** design approved, not yet implemented

A new `lc_classification_multisurvey_step/` that consumes the multisurvey
`feature_step` output, runs the ZTF BHRF (Squidward 2.1.0) classifier, and writes
probabilities into `multisurvey_ztf.probability` through `scribe_multisurvey`.

---

## 1. Why a new step

The deployed `lc_classification_step/` is a legacy, multi-survey, multi-model step:
it carries eight model families through `models_settings.configurator`, builds a
detections DTO the ZTF model never reads, and writes probabilities to the scribe
with **string** class names keyed by a **string** `_id`. The multisurvey database
keys probabilities by `(oid bigint, sid, classifier_id, class_id)` with an integer
`class_id` resolved from the `taxonomy` table.

Bridging those two inside one step means a config-switched fork through every
parser. The repo has already answered this shape question three times —
`correction_multisurvey_step`, `magstats_multisurvey_step`,
`stamp_classifier_2025_multisurvey_step`, `scribe_multisurvey` are all standalone
sibling directories. This follows that convention.

The offline pipeline in `feature_step/features/offline/` (in the `~/desktop/pipeline`
checkout, branch `fix/ztf-feature-parser-extra-fields`) already implements and
validates the BHRF → multisurvey-probability path end to end. **This step is the
streaming port of that offline logic**, not a new derivation. §5 tracks the port
line by line.

## 2. Decisions

| # | Decision | Chosen |
|---|---|---|
| 1 | Structure | **New `lc_classification_multisurvey_step/` dir**, copy-and-adapt. Legacy step untouched. |
| 2 | Model scope | **ZTF BHRF only** (`SquidwardFeaturesClassifier`). No elasticc/balto/messi/toretto/barney/mbappe/anomaly. A one-entry `models_settings.configurator` is kept so the model/mapper/path stay env-driven, matching the stamp step. |
| 3 | Persistence | **Scribe-only.** No DB writes from the step. `scribe_multisurvey` owns the upsert. |
| 4 | `class_name → class_id` | **Read-only DB lookup at startup**, cached. DB `taxonomy` is the authority. |
| 5 | Downstream Kafka topic | **New multisurvey output schema — placeholder for now.** Shape deferred. |

### Model and mapper live in `alerce_classifiers`

`alerce_classifiers` is a git submodule (`.gitmodules`: `alerce_classifiers` →
`alercebroker/alerce_classifiers`). `SquidwardFeaturesClassifier` and
`SquidwardMapper` live there. **The step owns no model code** — it resolves both
from config via `apf.core.get_class`, exactly as
`stamp_classifier_2025_multisurvey_step/step.py:83` does.

## 3. Module layout

```
lc_classification_multisurvey_step/
├── pyproject.toml
├── Dockerfile
├── README.md
├── .gitignore
├── settings.py                     # env → config dict
├── models_settings.py              # squidward_params + configurator (one entry)
├── credentials.py                  # DB creds (from correction_multisurvey_step)
├── scripts/
│   └── run_step.py
├── lc_classification_multisurvey_step/
│   ├── __init__.py
│   ├── step.py                     # LateClassifierMultisurvey(GenericStep)
│   ├── input_dto.py                # messages → features-only InputDTO
│   ├── probabilities.py            # OutputDTO → scribe-ready rows (5 heads)
│   ├── db/
│   │   ├── __init__.py
│   │   └── db.py                   # PSQLConnection + get_taxonomy_by_classifier_id
│   └── output_parser.py            # PLACEHOLDER downstream producer
└── tests/
    ├── __init__.py
    └── unittest/
        ├── test_probabilities.py
        ├── test_input_dto.py
        └── test_taxonomy.py
```

Model wiring in `step.__init__`:

```python
self.mapper = get_class(config["MODEL_CONFIG"]["CLASS_MAPPER"])()
self.model  = get_class(config["MODEL_CONFIG"]["CLASS"])(
    **{"mapper": self.mapper, **config["MODEL_CONFIG"]["PARAMS"]}
)
```

`models_settings.py` keeps a single `squidward_params` entry (mirroring the stamp
step's one-entry `configurator`) so the model class, mapper class and model path
stay env-driven rather than hardcoded:

```python
def squidward_params(model_class: str):
    return {
        "CLASS": model_class,
        "CLASS_MAPPER": os.getenv("CLASS_MAPPER"),
        "PARAMS": {"model_path": os.getenv("MODEL_PATH")},
        "NAME": model_class.split(".")[-1],
        "VERSION": os.getenv("MODEL_VERSION", "2.1.0"),
    }
```

## 4. Data flow

```
feature_step output topic
  │
  ├─ execute(messages)
  │    messages → input_dto.create_input_dto(messages)   # features-only
  │    model.can_predict(dto)  → skip batch if False
  │    model.predict(dto)      → OutputDTO (batched, multi-oid frames)
  │    probabilities.build_probability_rows(dto, lastmjd_by_oid, taxonomy_maps, ...)
  │      → for each of 5 heads: melt by oid → row dicts
  │
  ├─ post_execute → produce_scribe(rows)
  │    one `update-probability` command per row → scribe_multisurvey topic
  │
  └─ pre_produce → output_parser (PLACEHOLDER)
```

**Startup:** `get_taxonomy_by_classifier_id` for classifier ids 5–9 →
`{classifier_id: {class_name: class_id}}`, cached on the step. The connection is
opened once and not used again. Nothing else in the step touches the DB.

### `oid` is already a bigint

Unlike `stamp_classifier_2025_multisurvey_step` — which starts from raw ZTF alerts
and must call `idmapper.catalog_oid_to_masterid` — the multisurvey `feature_step`
already emits the bigint masterid (see `feature_step/features/utils/parsers.py:521`,
which does `int(oid)` on the same value). The Avro field is typed `string`, so the
step does `int(msg["oid"])` and calls no idmapper.

### Features-only InputDTO

`SquidwardFeaturesClassifier.can_predict` inspects only `input_dto.features`, and
`predict` calls `self.mapper.preprocess(input_dto)` which reads only features. So
detections / non-detections / xmatch / stamps are passed **empty**.

This is deliberate, and follows offline `classify.py`, which documents that it
"sidesteps `lc_classification_step`'s stale candid schema". The legacy
`create_detections_dto` builds a detections frame the model never reads, and
unpickles `extra_fields` bytes to do it. Dropping it removes the stale-schema
coupling and the pickle round-trip.

## 5. What is ported from offline, and what changes

The offline reference is `~/desktop/pipeline/feature_step/features/offline/`.

| Offline symbol | Port | Change |
|---|---|---|
| `probability_writer.CLASSIFIER_IDS` = `[5,6,7,8,9]` | verbatim | — |
| `probability_writer.classifier_version_to_smallint` | verbatim | `"2.1.0"` → `210`; strips a `_suffix` on the patch part |
| `probability_writer._iter_frames` | verbatim | the 5-head mapping; see §6 |
| `probability_writer.build_probability_rows` | **adapted** | offline is strictly per-oid and raises on a multi-row frame; the step is batched, so melt by oid instead |
| `probability_writer.write_probabilities` | **dropped** | scribe-only (decision 3) |
| `db.fetch_taxonomy_maps` | adapted | same query, via `PSQLConnection` instead of a raw engine |
| `classify.load_squidward_model` | **dropped** | `get_class` from config instead (§2) |
| `classify.features_message_to_dto` | adapted | one row per message instead of one oid |
| `classify.classify_astro_object` | partially | keep `can_predict` → empty-OutputDTO-on-false; drop `parse_output` (feature_step already ran it) |
| `classify._lc_lastmjd` | adapted | max MJD from the message's `detections` array, not from DB |
| `classifier_taxonomy_lut` | **not ported** | the fixture stays the seed authority in the offline repo; the step reads the DB |

The single most important piece the stamp step does **not** have is
`_iter_frames`: `stamp_classifier_2025_multisurvey_step` writes only
`output_dto.probabilities`, i.e. one head. BHRF emits five.

## 6. The five heads

```python
[
    (5, output_dto.probabilities),                        # flat, 21 leaves
    (6, hierarchical["top"]),                             # Periodic/Stochastic/Transient
    (7, hierarchical["children"]["Transient"]),           # 6 classes
    (8, hierarchical["children"]["Stochastic"]),          # 6 classes
    (9, hierarchical["children"]["Periodic"]),            # 9 classes
]
```

Classifier ids 5–9 correspond to `lc_classifier_BHRF_forced_phot{,_top,_transient,
_stochastic,_periodic}`, seeded in `multisurvey_ztf` (verified applied 2026-08-03).
Class names are md5-verified against the deployed pickle; the transient class is
**`SESN`**, not `SNIbc`.

A head whose frame is `None` or empty is skipped, not written.

## 7. Probability row contract

`scribe_multisurvey` already accepts exactly this payload — no scribe change is
needed. `parse_probability_table` (`scribe_multisurvey/sql_scribe/sql/command/parser.py:430`):

```
{oid, sid, classifier_id, classifier_version, class_id, probability, ranking, lastmjd}
```

Command envelope, matching `stamp_classifier_2025_multisurvey_step/step.py:272`
and accepted by `decode.command_factory`:

```json
{"step": "update-probability", "survey": "ztf", "payload": { ...row... }}
```

produced as `{"payload": json.dumps(command)}` with the Kafka key set to
`str(oid)`.

`ProbabilityCommand.db_operation`
(`scribe_multisurvey/sql_scribe/sql/command/commands.py:607`) dedups by
`(oid, sid, classifier_id, class_id)` keeping the highest `lastmjd`, then upserts
with `ON CONFLICT (pk_probability_oid_classifierid_classid) DO UPDATE` on
`probability`, `ranking`, `lastmjd`. That is the update-on-reclassify behaviour
this step needs — it matches offline `write_probabilities`, and is **not** the
`DO NOTHING` the stamp step uses.

Field derivations:

- `oid` — `int(msg["oid"])`.
- `sid` — from config `SID`, default `0` (ZTF).
- `classifier_id` — the head's id, 5–9.
- `classifier_version` — `classifier_version_to_smallint(MODEL_VERSION)` → `210`.
- `class_id` — `taxonomy_maps[classifier_id][class_name]`, exact string match.
- `probability` — the model's value.
- `ranking` — dense rank descending **within (oid, classifier_id)**, i.e. per head.
- `lastmjd` — `max(det["mjd"] for det in msg["detections"])`. Already MJD; **do not**
  subtract 2400000.5. The message's `detections` array carries forced photometry
  too (each entry has a `forced` flag), so this is the max over detections and
  forced together, matching offline `_lc_lastmjd`.

## 8. Error handling

`probability.class_id` is a foreign key. The stamp step's `class_name_to_id`
returns `-1` on a miss, which either violates the FK or stores a garbage class.
Offline instead raises. This step splits the difference by when the problem is
detectable:

- **Startup, fail fast.** After fetching the taxonomy, assert every classifier id
  in 5–9 has a non-empty map. If any is missing, raise and refuse to start — an
  unseeded or partially-seeded taxonomy is a deploy error, and a step that starts
  anyway would silently drop every probability it produces.
- **Per batch, skip and log.** If a class name coming out of the model is absent
  from its head's map, log an error identifying the oid, classifier id and class
  name, and drop **that oid's rows for that head**. Do not emit `class_id = -1`,
  and do not kill the batch — one model/taxonomy drift should not stop the
  consumer.
- **`can_predict` false** → produce nothing, log, return an empty OutputDTO,
  matching offline `classify_astro_object` and the legacy step.
- **Messages with no features** (`msg["features"]` is `None`) are filtered out
  before the DTO is built.

## 9. Placeholder downstream output

Decision 5 is "new multisurvey output schema", but its shape is deferred. For now:

- `output_parser.py` holds a `MultisurveyOutputParser` whose `parse` returns a
  minimal per-oid message — `{oid, classifier_name, classifier_version}` plus the
  top-ranked class per head — and is marked `# PLACEHOLDER` with a pointer to this
  section.
- No `schemas/lc_classification_multisurvey_step/` Avro file is added yet.
- The producer is configured but the shape is not treated as a contract; nothing
  downstream should be pointed at it until the schema is designed.

Deferring this is safe because decision 3 makes the scribe the real output path.

## 10. Configuration

| Env var | Purpose | Default |
|---|---|---|
| `CONSUMER_TOPICS`, `CONSUMER_SERVER`, `CONSUMER_GROUP_ID` | feature_step output topic | required |
| `MODEL_CLASS` | `alerce_classifiers.squidward.model.SquidwardFeaturesClassifier` | required |
| `CLASS_MAPPER` | `alerce_classifiers.squidward.mapper.SquidwardMapper` | required |
| `MODEL_PATH` | S3 url of `squidward/2.1.0/hierarchical_random_forest_model.pkl` | required |
| `MODEL_VERSION` | version string → smallint | `2.1.0` |
| `SCRIBE_SERVER`, `SCRIBE_TOPIC` | scribe_multisurvey topic | required |
| `PSQL_*` / `SCHEMA` | startup taxonomy read | required, schema `multisurvey_ztf` |
| `SID` | survey id written into `probability.sid` | `0` |
| `MIN_DETECTIONS` | optional pre-classification filter (§13) | unset |

Classifier ids 5–9 are **not** configurable. They are pinned constants matching the
seeded `classifier` rows, and the head→id mapping in §6 is positional; making them
an env var would let the two drift.

## 11. Testing

Pure-function unit tests, following the offline test layout
(`tests/unittest/test_offline_*.py`):

- `test_probabilities.py` — the 5-head split; melt-by-oid over a multi-oid frame;
  per-head dense ranking; version → smallint; a missing/empty head is skipped;
  an unknown class name drops that head's rows for that oid and logs rather than
  raising; `lastmjd` taken as the max over detections without JD subtraction.
- `test_input_dto.py` — features-only DTO; `oid` cast to int; messages with
  `features: None` filtered; empty batch → empty DTO.
- `test_taxonomy.py` — `get_taxonomy_by_classifier_id` map construction against a
  mocked session; startup assertion fires when a head's map is empty.

**Equivalence test against the offline reference.** The step's row builder and
offline `probability_writer.build_probability_rows` must agree. For a handful of
real OIDs, run the offline classifier, feed the same `OutputDTO` through both row
builders, and assert the row sets are identical modulo ordering. This is the test
that actually protects the port — the unit tests above only check the port's
internal consistency. It needs the `alerce_classifiers` submodule initialised and
`MODEL_PATH` set, so it is marked as an opt-in integration test, not part of the
default unit run.

## 12. Out of scope

- Changes to `scribe_multisurvey` — its `ProbabilityCommand` already fits.
- Changes to the legacy `lc_classification_step`.
- The downstream output schema (§9).
- Seeding `classifier` / `taxonomy` rows — already applied to live `multisurvey_ztf`.
- Back-porting the seeds to the db-plugins authority file — tracked separately in
  the offline repo's `FLOW.md` §7.
- LSST / Rubin models.
- Helm chart (`charts/lc_classification_multisurvey_step/`) — deployment is a
  follow-up once the step runs locally.

## 13. Open questions

- **`MIN_DETECTIONS`.** The legacy step filters objects below a detection count
  before classifying. Since features are already computed upstream, it is not
  obvious this step should re-filter. **Resolved for now:** carry the config knob,
  default it to unset — every message that has features gets classified. Counts
  non-forced detections only, as the legacy step does. Revisit if the model
  produces junk on sparse light curves.
- **Batch size vs. `classify_batch`.** The model classifies the whole batch in one
  call. Whether the feature_step consume batch size is a good predict batch size
  has not been measured.
