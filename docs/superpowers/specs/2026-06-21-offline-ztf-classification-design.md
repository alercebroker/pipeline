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
  classifier-input message with a wide `features` dict.
- `lc_classification.core.parsers.input_dto.create_input_dto` (lc_classification) —
  message list → `InputDTO`.
- `alerce_classifiers.squidward.model.SquidwardFeaturesClassifier` + `SquidwardMapper`
  — the real model.

The only new code is glue: load the model from env vars, and stitch the existing
real functions together.

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
              create_input_dto([out_message])                         [REAL lc_classification code]
                                     │  InputDTO
                                     ▼
        SquidwardFeaturesClassifier(env config).predict              [REAL model, BHRF 2.1.0]
                                     │
                                     ▼
                       OutputDTO → probabilities (top + hierarchical)
```

## Environment decision

End-to-end requires one environment with all of: `features` (offline module + parser),
`lc_classifier`, `alerce_classifiers[ztf]`, and `lc_classification`. Today neither
step's env has all four:
- `feature_step` has `lc_classifier` but not `alerce_classifiers`/`lc_classification`.
- `lc_classification_step` has `alerce_classifiers[ztf]` but not `features`.

**Decision:** keep the offline code in `feature_step/features/offline/` (next to the
feature code) and extend the **feature_step** poetry env with the two missing path deps.
Rationale: end-to-end is "features → classify", the feature code already lives here, and
reusing `parse_output` is local.

Note: importing `lc_classification.core.parsers.input_dto` triggers
`lc_classification.core.__init__` (`from .step import *`), which pulls the full step
import chain (apf, kafka, alerce_classifiers). This is acceptable — feature_step's env
already has `apf` (its own step is a `GenericStep`), and we are adding
`alerce_classifiers[ztf]` anyway. `models_settings.py`/`settings.py` are top-level
scripts in `lc_classification_step`, **not** part of the installed `lc_classification`
package, so they are not importable; the small env-var read is inlined instead.

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
| `feature_step/pyproject.toml` | Add path deps `alerce_classifiers = {path="../alerce_classifiers", develop=true, extras=["ztf"]}` and `lc_classification = {path="../lc_classification_step", develop=true}`. Re-lock. |
| `features/offline/classify.py` | **New.** `load_squidward_model()` → `(model, name, version)` from env vars; `classify_astro_object(ao, message, model)` → out_message via `parse_output` → `create_input_dto` → `can_predict` + `predict` → `OutputDTO`; `classify_oid(...)` convenience glue (DB → ao → classify). |
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
