# `features.offline` — offline ZTF feature computation

Batch / standalone reproduction of the ZTF **feature_step**, computed directly
from the database instead of from a Kafka stream. It builds the same
**magstats_ms_step ZTF output message** the magstats step emits, then runs the
**real** feature_step parser + `lc_classifier` extractor on it.

Use it to (re)compute ZTF light-curve features for arbitrary objects offline —
e.g. to generate a feature dataset, backfill, or validate against stored
features — without standing up the streaming step.

> This is **not vendored code**. It imports the live `features.utils.parsers`,
> `lc_classifier`, and `idmapper` from this repo, so it always tracks the current
> pipeline behavior.

## Pipeline flow vs. here

The production feature_step is a Kafka `GenericStep`:

```
consume message → discard_bogus_detections → detections_to_astro_object
                → preprocess → extract → produce to scribe/output
```

This package keeps the **pure computation** and drops the streaming plumbing:

```
multisurvey_ztf DB ──db.fetch_*──▶ build_message  (magstats_ms_step ZTF output message)
                                     │
   fetch_references ─┐               ▼
   fetch_allwise ────┴──▶ compute_features
                          ├─ discard_bogus_detections   (lc_classifier)
                          ├─ detections_to_astro_object  (features.utils.parsers, forced=[])
                          ├─ ZTFLightcurvePreprocessor   (lc_classifier)
                          └─ ZTFFeatureExtractor         (lc_classifier)
                                     │
                                     ▼
                          long features DataFrame  [name, value, fid, sid, version]
```

AllWISE and reference data are **not** in the message (the magstats schema has no
xmatch field); they're fetched separately from the DB and passed alongside.

## Modules

| File | What it does |
|------|--------------|
| `db.py` | DB readers (SQLAlchemy). `fetch_detections`, `fetch_forced_photometry`, `fetch_ps1`, `fetch_allwise`, `fetch_references`; plus `fetch_alerce_features` / `list_alerce_feature_versions` for the legacy `alerce.feature` reference. Constants: `SCHEMA="multisurvey_ztf"` (env-overridable via `OFFLINE_DB_SCHEMA`), `SID=0` (ZTF), `ALLWISE_CATID=0`, `ALERCE_SCHEMA="alerce"`. |
| `message.py` | `build_message(oid, detections, forced, ps1)` → the magstats_ms_step ZTF output message dict (schema `schemas/magstats_ms_step/ztf/output.avsc`, the `magstats_ms_ztf` record). Forced epochs are emitted **inline** in `detections` with `forced=True`; per-epoch aux (rb / procstatus / reference / PS1) go in `extra_fields`. |
| `lc_features.py` | The assembly layer. `compute_features(message, references_db, allwise, min_detections=1, preprocessor=None, extractor=None)` → features DataFrame, or `None` if too few real detections. Helpers: `_prepare_detections` (drop bogus, enforce min, add `aid`/`index_column`), `_xmatches` (AllWISE → the shape the parser reads), `message_to_astro_object`. Also `compute_db_features(...)` → DB-ready rows `[oid, sid, feature_id, band, version, value]` (drop NaN + `fid→band` + `name→feature_id` via the fixture LUT), following the production save rules. |
| `feature_lut.py` | Local ZTF `feature_name_lut`/`feature_version_lut` fixture (the DB ones are empty) + loaders: `load_feature_name_lut`, `version_name_to_id`, `default_version_name`. |
| `feature_writer.py` | `write_features(rows, credentials, schema, execute=False)` upserts the DB-ready rows into `<schema>.feature` (`ON CONFLICT … DO UPDATE`); dry-run unless `execute=True`. |
| `feature_compare.py` | Pure diff utilities. `compare_feature_frames(ours, theirs, rtol, atol)` → `(merged, summary)` classifying each (name, fid) as match / differ / only_ours / only_theirs. `latest_feature_version(versions)` picks the newest modern version string. |
| `classify.py` | Classification bridge. `load_squidward_model()` builds the BHRF `SquidwardFeaturesClassifier` from env vars (`MODEL_PATH`, `MAPPER_CLASS`); `classify_astro_object(ao, message, model)` names features via the real `parse_output`, builds a **features-only** `InputDTO` (`input_dto_factory`), and runs `can_predict`+`predict`; `classify_oid(...)` is the DB->probabilities convenience path. The Squidward model reads only `features`, so detections are passed empty (no `lc_classification` dependency). |

## Scripts (`feature_step/scripts/`)

| Script | Purpose |
|--------|---------|
| `offline_compute_features.py --oid <bigint> [--credentials PATH] [--save [--execute] [--write-credentials PATH]]` | DB → message → features for one oid; prints a populated long frame. Add `--save` to upsert into `<schema>.feature` (dry-run unless `--execute` is also given; `--write-credentials` supplies write-capable DB credentials). |
| `offline_benchmark_features.py [--n N] [--warmup K] [--min-det M] [--credentials PATH]` | Times `compute_features` over N real oids (extractor built once, warm-up excluded). |
| `offline_compare_vs_alerce.py --oid <bigint|ZTFstr> [--version V] [--rtol] [--atol]` | Runs our pipeline on a multisurvey oid, maps it to its ZTF string oid via `idmapper`, fetches `alerce.feature` for that object, and diffs. |
| `offline_classify.py --oid <bigint> [--credentials PATH] [--min-det M]` | DB -> message -> features -> BHRF probabilities for one oid. Requires `MODEL_PATH` env (the S3 BHRF url). |
| `offline_compare_probabilities.py` | **Deferred** — predicted vs. stored probabilities. Pending the stored-probability table. |

## Running

Use the pipeline's environment (so `apf` / `confluent_kafka` / `lc_classifier` /
`idmapper` are present — importing `features` triggers `features/__init__` which
imports the streaming `step`):

```bash
cd feature_step
poetry install
poetry run python scripts/offline_compute_features.py --oid 36028941624528297 --credentials /path/to/credentials.json
```

`--credentials` is a JSON with the DB connection (same shape the other steps use).

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

## Fixes this relies on

The ZTF feature path in `features.utils.parsers` and `lc_classifier` had to be made
consistent with the magstats_ms_step ZTF output message contract (these are committed on the
same branch as this package):

- **`features/utils/parsers.py`** — `detections_to_astro_object` reads `mag_corr` /
  `e_mag_corr_ext` (not `magpsf_corr` / `sigmapsf_corr_ext`), reads aux fields
  (`rb`, `distnr`, `rfid`, `procstatus`, `sharpnr`, `chinr`, PS1) from
  `extra_fields`, and routes forced epochs via the per-row `forced` flag (the
  message carries them inline; the `forced` arg must be empty).
- **`lc_classifier/.../core/base.py`** — `discard_bogus_detections` coerces
  `procstatus` to `str` before the `"0"` / `"57"` comparison, so valid forced
  epochs with an integer `procstatus` aren't dropped.

## Validating against stored features

Stored ZTF features live in the legacy **`alerce.feature`** table (string oids,
multiple stacked versions, no compute timestamp); `multisurvey.feature` currently
holds **no** ZTF rows (LSST only). The multisurvey↔alerce oid relation is the
deterministic `idmapper` encoding (ZTF string ↔ bigint), so `offline_compare_vs_alerce.py`
maps between them automatically.

**Status:** the pipeline produces correct, populated features on real objects,
now reading the full **`multisurvey_ztf`** dataset (the earlier ~3-month
`multisurvey` slice is no longer used). The truncated-light-curve blocker on a
value-level equality check vs `alerce.feature` is therefore lifted; pass
`--version` matching the deployed `lc_classifier` for an apples-to-apples
comparison (not yet confirmed on the new data).
