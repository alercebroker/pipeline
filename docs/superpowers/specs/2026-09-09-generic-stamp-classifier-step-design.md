# Generic stamp classifier step (Rubin)

Goal: run any stamp classifier defined in `alerce_classifiers` (first case: the
stamp hunter) through `rubin_stamp_classifier_step` by changing config only.
The step keeps the LSST message adapter; everything model-specific moves to
the model package or to config.

Scope: `rubin_stamp_classifier_step` and a new package in `alerce_classifiers`.
The ZTF stamp steps are legacy and are not touched.

## What stays as is

- `pre_execute`: the LSST adapter. Extracts a superset of diaSource fields
  plus the three cutouts. Models pick `stamps_cols` and `order_features`
  from their own hparams.
- Output schema: `probabilities` is a `map<string,double>`, taxonomy-agnostic.
- DB writer, scribe producer, ranking, and taxonomy lookup: all keyed on
  `CLS_ID` and class names.
- Model contract: `AlerceModel` with `predict(InputDTO) -> OutputDTO`,
  probabilities indexed by `diaObjectId`, plus `model_version`.

## Changes

### 1. Model loading is config-driven

Today: `step.py` imports `alerce_classifiers.rubin.StampClassifierModel`
directly and builds it with `model_path` only.

Proposed: same pattern as the lc classification steps.

```yaml
MODEL_CONFIG:
  CLASS: alerce_classifiers.rubin.StampClassifierModel
  PARAMS:
    model_path: https://.../1.0.0/model.zip
  CLS_ID: 3
```

The step does `get_class(CLASS)(**PARAMS)`. Hunter is another `CLASS`.

### 2. Asteroid fallback is derived from the taxonomy

Today: `execute` hardcodes `{AGN, SN, VS, asteroid, bogus}` with
`asteroid: 1.0` for `ssObjectId` alerts.

Proposed: build the fallback from the taxonomy already loaded by `CLS_ID`
(or from the model's class list): every class at 0.0, the asteroid class at
1.0. The asteroid class name is `MODEL_CONFIG.ASTEROID_CLASS`, defaulting to
`asteroid`. If the taxonomy has no such class the step fails at startup
rather than writing `class_id = -1` rows.

### 3. Canonical stamp column names, no rename flag

Today: `RENAME_STAMP_COLUMNS` renames `visit_image` to `flux_Science_data`
and so on, to match one trained model's `stamps_cols`.

Proposed: the step always emits `science`, `reference`, `difference`. Each
model in `alerce_classifiers` maps those to whatever its network expects
inside its own mapper. The flag is removed. The existing rubin model gets
the mapping in its mapper so current deployments keep working.

### 4. Model version comes from the model, robustly

Today: `_get_model_version` returns `model_path.split("/")[-2]`, which only
works for `.../<version>/<file>.zip` URLs.

Proposed: keep the URL convention as the default but let `MODEL_CONFIG.VERSION`
override it. The step reads `self.model.model_version` as now.

### 5. Metadata superset is extended only when a model needs it

If hunter needs a diaSource field the adapter does not extract, add it to
the superset in `pre_execute` with the same placeholder policy used for
`airmass`, `magLim`, `seeing`. Models that do not list it ignore it.

### 6. Dependencies and image

- `alerce_classifiers`: new package `alerce_classifiers/hunter/` with
  `model.py`, `mapper.py`, `arch.py`, plus a `hunter` extra in
  `pyproject.toml`. It may reuse the padding and normalization helpers in
  `alerce_classifiers/rubin/mapper.py`.
- Step `pyproject.toml`: drop the leftover `stamp_full` extra from main deps.
  Add a `hunter` group mirroring the `rubin` group, or fold both into one
  group if the deps are identical.
- Dockerfile: install the group(s) needed; one image serves both models.

### 7. Tests

Integration tests parametrize `MODEL_CONFIG` over the two models. Each
model needs a fixture model directory, taxonomy rows for its `CLS_ID`, and
an assertion that the produced `probabilities` keys equal the taxonomy.

## Deployment

One deployment per model: separate `CLS_ID`, consumer group, output topic,
and `MODEL_CONFIG`. Nothing in the step assumes a single classifier per
survey.

## Open points

- Hunter inputs: confirm it uses the three cutouts and a subset of the
  fields already extracted, or list what is missing (change 5).
- Hunter model artifact layout: `model.keras` plus `hparams.yaml` inside a
  zip, as rubin does, or something else the loader must handle.
