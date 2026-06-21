# Offline ZTF classification — design

**Date:** 2026-06-21
**Branch:** `fix/ztf-feature-parser-extra-fields`
**Status:** approved (brainstorming) → ready for implementation plan

## Goal

Extend the existing offline ZTF feature tooling (`feature_step/features/offline/`)
so it can run the **BHRF light-curve classifier** (`SquidwardFeaturesClassifier`,
`lc_classifier_BHRF_forced_phot`, version 2.1.0) end-to-end and offline:

```
oid → DB → correction-ztf message → features → BHRF model → probabilities
```

A compare-vs-stored-probabilities tool (analogous to `offline_compare_vs_alerce.py`
for features) is **in scope conceptually but deferred** — the stored-probability
DB table is not yet pinned down. It ships as a documented stub.

## Principles

Same as the existing offline module: **not vendored**. Reuse the live pipeline
functions so behavior tracks production:
- `features.utils.parsers.parse_output` (feature_step) — long features frame → the
  classifier-input message with a wide, band-suffixed `features` dict (these are the
  exact feature names the model's 199-feature list expects).
- `alerce_classifiers.base.factories.input_dto_factory` (alerce_classifiers) — build
  the `InputDTO` directly.
- `alerce_classifiers.squidward.model.SquidwardFeaturesClassifier` + `SquidwardMapper`
  — the real model.

The only new code is glue: load the model from env vars, and stitch the existing
real functions together.

### Why not `create_input_dto` / `lc_classification` (planning finding)

The original plan was to reuse `lc_classification.core.parsers.input_dto.create_input_dto`.
Planning revealed this is both unnecessary and broken:

1. **The Squidward model is features-only.** `SquidwardFeaturesClassifier.can_predict`
   / `predict` and `SquidwardMapper.preprocess` read **only `input_dto.features`** —
   detections are never touched.
2. **The local `lc_classification_step` is on a stale schema.** Its
   `create_detections_dto` does `drop_duplicates(["candid", "oid"])`, but v11
   correction-ztf messages carry `measurement_id`, not `candid`, so it would
   `KeyError`. (It also pulls `apf`/kafka via `lc_classification/core/__init__.py`'s
   `from .step import *`.)

So we **drop the `lc_classification` dependency entirely** and build a features-only
`InputDTO` via the real `alerce_classifiers` factory, passing empty DataFrames for
detections/non_detections/xmatch/stamps (the DTO containers do no validation).

## Architecture / data flow

```
multisurvey DB ──db.fetch_*──▶ build_message (correction-ztf)        [existing]
                                     │
                                     ▼
   references_db, allwise ─▶ message_to_astro_object                  [existing, real parser]
                                     │  AstroObject
                                     ▼
                          preprocess + extract                        [existing compute path]
                                     │
                                     ▼
              parse_output([ao],[msg],candids)                        [REAL feature_step code]
                                     │  classifier-input message (wide `features` dict)
                                     ▼
              wide features_df (one row, indexed by oid)
                                     │
                                     ▼
              input_dto_factory(empty,empty,features_df,empty,empty)  [REAL alerce_classifiers code]
                                     │  InputDTO (features-only)
                                     ▼
        SquidwardFeaturesClassifier(env config).predict              [REAL model, BHRF 2.1.0]
                                     │
                                     ▼
                       OutputDTO → probabilities (top + hierarchical)
```

## Environment decision

End-to-end requires one environment with: `features` (offline module + parser),
`lc_classifier`, and `alerce_classifiers[ztf]`. Today:
- `feature_step` has `lc_classifier` but not `alerce_classifiers`.
- `lc_classification_step` has `alerce_classifiers[ztf]` but not `features`.

**Decision:** keep the offline code in `feature_step/features/offline/` (next to the
feature code) and extend the **feature_step** poetry env with **only**
`alerce_classifiers[ztf]`. Rationale: end-to-end is "features → classify", the feature
code already lives here, reusing `parse_output` is local, and (per the planning finding
above) the `lc_classification` package is not needed.

**Prerequisite:** the `alerce_classifiers` git submodule must be initialized
(`git submodule update --init alerce_classifiers`) — done during planning.

`models_settings.py`/`settings.py` are top-level scripts in `lc_classification_step`,
**not** part of any installed package, so they are not importable; the small env-var
read is inlined instead.

## Model config (env-driven, mirrors deployment)

Read the same env vars the deployed step uses:
- `MODEL_PATH` — the model pickle URL/path (e.g. the S3 BHRF 2.1.0 url).
- `MAPPER_CLASS` — `alerce_classifiers.squidward.mapper.SquidwardMapper`.
- `CLASSIFIER_NAME` (optional) — defaults to the class name; deployment tags rows
  `lc_classifier_BHRF_forced_phot`, which matters only for output labeling / the
  deferred compare, so it is overridable but not required now.

Instantiate exactly as `step.py` does for non-ZTF models: instantiate the mapper,
then `SquidwardFeaturesClassifier(model_path=..., mapper=mapper_instance)`. The
classifier version comes from `model.model_version` (same as the step).

## Components

| Item | Change |
|---|---|
| `feature_step/pyproject.toml` | Add path dep `alerce_classifiers = {path="../alerce_classifiers", develop=true, extras=["ztf"]}`. Re-lock. |
| `features/offline/classify.py` | **New.** `load_squidward_model()` → `(model, name, version)` from env vars; `features_message_to_dto(out_message)` → features-only `InputDTO` via `input_dto_factory`; `classify_astro_object(ao, message, model)` → out_message via `parse_output` → DTO → `can_predict` + `predict` → `OutputDTO`; `classify_oid(...)` convenience glue (DB → ao → classify). |
| `features/offline/lc_features.py` | Small refactor: expose the `AstroObject` (so classify reuses the same ao without recomputing). `compute_features` keeps its current return signature for back-compat. |
| `feature_step/scripts/offline_classify.py` | **New.** `--oid <bigint> [--credentials PATH] [--min-det M]` → prints top + per-class probabilities. |
| `features/offline/db.py` | **Deferred stub.** `fetch_stored_probabilities(...)` raising `NotImplementedError("pending: stored-probability table TBD")`. |
| `feature_step/scripts/offline_compare_probabilities.py` | **Deferred skeleton.** Errors with the same "pending" message; wired up once the table is known. |
| `features/offline/README.md` | Add a "Classification" section + the new script rows; note the compare tool is pending. |

## Error handling

- Too few real detections → `message_to_astro_object` already returns `None`; the
  classify path returns empty probabilities (mirrors the step's "can't predict" path).
- `model.can_predict(dto)` false → return empty `OutputDTO`, log the reason (mirrors
  `step.execute`).
- Missing `MODEL_PATH`/`MAPPER_CLASS` → fail fast with a clear message.

## Testing

- **Unit:** `classify_astro_object` with a fake model (stub `can_predict`/`predict`),
  asserting the out_message → DTO bridge is correct. Mirrors how the step's tests
  inject a fake model. No network / model download.
- **Manual / integration:** run `offline_classify.py` on a real oid with `MODEL_PATH`
  set to the S3 BHRF url; confirm a populated probability frame (top + hierarchical).

## Out of scope / deferred

- Compare predicted vs. stored probabilities (needs the stored-probability table
  name, schema, and classifier_name/version filter). Ships as a stub now.
- Batch/benchmark classification variants (can follow the feature equivalents later).
