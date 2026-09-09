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
ranks first as SN. The ss branch in `execute` stays as it is and is dead
code for hunter.

## Changes

### 1. Model loading is config-driven

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

### 2. SN forwarder in the rubin deployment

Today: one producer, `rubin_stamp_classifier` topic, probabilities only.

Proposed: an optional second producer, `SN_FORWARD_PRODUCER_CONFIG`, with
the LSST alert schema as `SCHEMA_PATH` and `sn_candidates` as topic. When
configured, after prediction the step forwards the raw consumed alert for
every object whose ranking-1 class is `SN_FORWARD_CLASS` (default `SN`).
When absent, nothing changes. The hunter deployment does not set it.

`pre_execute` discards the raw message today. It needs to keep it alongside
the processed fields (keyed by `diaSourceId`) so the forwarder can re-emit
it without re-serializing from the DTO.

### 3. Stamp column names

Today: `RENAME_STAMP_COLUMNS` renames `visit_image` to `flux_Science_data`
and so on, to match the trained rubin model's `stamps_cols`.

Proposed: the hunter model's mapper accepts the step's names
(`visit_image`, `difference_image`, `reference_image`) directly, so the
hunter deployment runs with the flag off. The flag stays for rubin until
its model is retrained or its mapper is updated.

### 4. Model version

Today: `_get_model_version` returns `model_path.split("/")[-2]`.

Proposed: keep the URL convention as default, allow `MODEL_CONFIG.VERSION`
to override it. The hunter model zip should follow `.../<version>/<file>.zip`
anyway.

### 5. Dependencies and image

- `alerce_classifiers`: new package `alerce_classifiers/hunter/` with
  `model.py`, `mapper.py`, `arch.py`, and a `hunter` extra. It may reuse the
  padding and normalization helpers in `alerce_classifiers/rubin/mapper.py`.
- Step `pyproject.toml`: drop the leftover `stamp_full` extra from main deps.
  Add a `hunter` group mirroring `rubin`, or one group if the deps match.
- Dockerfile: install both groups; one image serves both deployments.

### 6. Tests

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
