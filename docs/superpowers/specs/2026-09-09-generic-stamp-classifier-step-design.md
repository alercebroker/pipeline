# Stamp classifier step: one step, two deployments (rubin + hunter)

Goal: run the stamp hunter classifier as a second deployment of
`rubin_stamp_classifier_step`, downstream of the rubin stamp classifier.
The hunter model lives in `alerce_classifiers` and is selected by config.

## Pipeline shape

```
lsst alerts -> stamp step [rubin model] -> rubin_stamp_classifier topic (all objects)
                                        -> sn_candidates topic (raw alert, ranking-1 SN only)
sn_candidates -> stamp step [hunter model] -> hunter output topic + DB + scribe
```

The `sn_candidates` topic carries the LSST alert unchanged, same schema as
the input. The hunter deployment therefore consumes exactly what the rubin
deployment consumes. `pre_execute`, the DTO, the DB writer, the scribe
producer, and the output schema are shared as they are.

ssObjectId alerts can never reach the hunter deployment: an asteroid never
ranks first as SN. The step still must not carry the asteroid rule, so it
moves to the rubin model (change 2).

## Changes

### 1. Model loading is config-driven

Status: implemented 2026-09-10 (step, integration test configs, README, chart values; unit tests in `tests/unit/test_model_loading.py`).

Today: `step.py` imports `alerce_classifiers.rubin.StampClassifierModel`
directly and builds it with `model_path` only.

Proposed: same pattern as the lc classification steps.

```yaml
MODEL_CONFIG:
  CLASS: alerce_classifiers.rubin.StampClassifierModel   # or the hunter class
  PARAMS:
    model_path: https://.../1.0.0/model.zip
  CLS_ID: 3
```

The step does `get_class(CLASS)(**PARAMS)`.

### 2. The asteroid rule moves to the rubin model

Status: implemented 2026-09-10 (step, db/scribe formatters, `pre_produce`, and
`alerce_classifiers.rubin.StampClassifierModel.predict`; unit tests in both repos).

Today: `execute` splits alerts into diaObject and ssObject lists, runs the
model on the first, and hardcodes `{AGN, SN, VS, asteroid, bogus}` with
`asteroid: 1.0` for the second. `db.py` and `_format_scribe_records`
re-derive `oid` and `sid` from the two ids.

Proposed: the step knows LSST identity and nothing about asteroids.

- `pre_execute` resolves identity once per alert: `oid` is `diaObjectId`
  or `ssObjectId`, `sid` is 1 or 2. Both go into the processed message and
  `sid` goes into the `Features` frame of the DTO.
- `execute` passes every alert to `model.predict` and emits one output
  message per row the model returns. No split, no class names in the step.
- `db.py` and the scribe formatter read `oid` and `sid` from the message.
- `alerce_classifiers.rubin.StampClassifierModel.predict` returns asteroid
  rows for `sid == 2` (every class at 0.0, `asteroid` at 1.0, from its
  own `dict_mapping_classes`) and runs the network only on `sid == 1`.
- The hunter model has no `sid` logic. It only ever receives `sid == 1`.

Rubin output does not change: same rows, same probabilities, same topic.

### 3. SN forwarder in the rubin deployment

Status: implemented 2026-09-14. `pre_execute` keeps the raw alert on the
message under `alert`, `execute` copies it to the output, `post_execute`
forwards after the scribe, `pre_produce` strips it. The forward producer is
the apf `KafkaProducer` with the LSST alert schema, so `sn_candidates` is in
apf container format and the hunter deployment consumes it with the plain
`apf.consumers.KafkaConsumer`. apf drains the producer before the offset
commit. Unit tests in `tests/unit/test_sn_forwarding.py`; the Kafka
integration test also checks the forward topic.

Today: one producer, `rubin_stamp_classifier` topic, probabilities only.

Proposed: an optional second producer, `SN_FORWARD_PRODUCER_CONFIG`, with
the LSST alert schema as `SCHEMA_PATH` and `sn_candidates` as topic. When
configured, after prediction the step forwards the raw consumed alert for
every object whose ranking-1 class is `SN_FORWARD_CLASS` (default `SN`).
When absent, nothing changes. The hunter deployment does not set it.

`pre_execute` discards the raw message today. It needs to keep it alongside
the processed fields (keyed by `diaSourceId`) so the forwarder can re-emit
it without re-serializing from the DTO.

### 4. Stamp column names

Status: implemented 2026-09-16. `alerce_classifiers.hunter.mapper` reads
`visit_image`, `reference_image`, `difference_image` directly (science,
template, difference order), so the hunter deployment runs with
`RENAME_STAMP_COLUMNS: false`.

Today: `RENAME_STAMP_COLUMNS` renames `visit_image` to `flux_Science_data`
and so on, to match the trained rubin model's `stamps_cols`.

Proposed: the hunter model's mapper accepts the step's names
(`visit_image`, `difference_image`, `reference_image`) directly, so the
hunter deployment runs with the flag off. The flag stays for rubin until
its model is retrained or its mapper is updated.

### 5. Model version

Status: implemented 2026-09-10. The override key is the existing top-level
`MODEL_VERSION` (already in every config and read the same way by the lc
classification step), not `MODEL_CONFIG.VERSION`; an empty string means
"use the model's". Unit tests in `tests/unit/test_model_version.py`.

Today: `_get_model_version` returns `model_path.split("/")[-2]`.

Proposed: keep the URL convention as default, allow `MODEL_CONFIG.VERSION`
to override it. The hunter model zip should follow `.../<version>/<file>.zip`
anyway.

### 6. Dependencies and image

Status: implemented 2026-09-16. `alerce_classifiers/hunter/` (`arch.py`,
`mapper.py`, `model.py`) is a PyTorch port of alerce-hunter-classifier and
loads its `best_model.pt` (state_dict + hparams + training config) as is;
class names are hardcoded in the model (`not_candidate`, `candidate`) since
the artifact carries none. The `hunter` extra (torch, numpy) is added to the
step's existing `rubin` dependency group, so one image serves both
deployments; `stamp_full` in main deps is still to be dropped. Unit tests in
`alerce_classifiers/tests/unit/test_hunter_{mapper,model}.py`.

- `alerce_classifiers`: new package `alerce_classifiers/hunter/` with
  `model.py`, `mapper.py`, `arch.py`, and a `hunter` extra. It may reuse the
  padding and normalization helpers in `alerce_classifiers/rubin/mapper.py`.
- Step `pyproject.toml`: drop the leftover `stamp_full` extra from main deps.
  Add a `hunter` group mirroring `rubin`, or one group if the deps match.
- Dockerfile: install both groups; one image serves both deployments.

### 7. Tests

Integration tests parametrize `MODEL_CONFIG` over the two models. Add a
test that the rubin deployment forwards ranking-1 SN alerts unchanged to
`sn_candidates` and forwards nothing else.

## Deployment

Two deployments of the same image: separate consumer group, input topic,
output topic, `CLS_ID`, and `MODEL_CONFIG`. Taxonomy rows for the hunter
`CLS_ID` must exist in the DB before the first run.

## Open points

- Hunter model artifact layout: `model.keras` plus `hparams.yaml` inside a
  zip, as rubin does, or something the loader must handle differently.
- Hunter metadata: confirm it uses a subset of the diaSource fields the
  adapter already extracts. If not, extend the superset in `pre_execute`
  with the same placeholder policy used for `airmass`, `magLim`, `seeing`.
