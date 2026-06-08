# `features.offline` — offline ZTF feature computation

Batch / standalone reproduction of the ZTF **feature_step**, computed directly
from the database instead of from a Kafka stream. It builds the same
`correction-ztf` message the magstats step emits, then runs the **real**
feature_step parser + `lc_classifier` extractor on it.

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
multisurvey DB ──db.fetch_*──▶ build_message  (correction-ztf message)
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
| `db.py` | DB readers (SQLAlchemy). `fetch_detections`, `fetch_forced_photometry`, `fetch_ps1`, `fetch_allwise`, `fetch_references`; plus `fetch_alerce_features` / `list_alerce_feature_versions` for the legacy `alerce.feature` reference. Constants: `SCHEMA="multisurvey"`, `SID=0` (ZTF), `ALLWISE_CATID=0`, `ALERCE_SCHEMA="alerce"`. |
| `message.py` | `build_message(oid, detections, forced, ps1)` → the `correction-ztf` message dict (conforms to the magstats_ms_step ZTF output schema, `ztf_correction.avsc`). Forced epochs are emitted **inline** in `detections` with `forced=True`; per-epoch aux (rb / procstatus / reference / PS1) go in `extra_fields`. |
| `lc_features.py` | The assembly layer. `compute_features(message, references_db, allwise, min_detections=1, preprocessor=None, extractor=None)` → features DataFrame, or `None` if too few real detections. Helpers: `_prepare_detections` (drop bogus, enforce min, add `aid`/`index_column`), `_xmatches` (AllWISE → the shape the parser reads), `message_to_astro_object`. |
| `feature_compare.py` | Pure diff utilities. `compare_feature_frames(ours, theirs, rtol, atol)` → `(merged, summary)` classifying each (name, fid) as match / differ / only_ours / only_theirs. `latest_feature_version(versions)` picks the newest modern version string. |

## Scripts (`feature_step/scripts/`)

| Script | Purpose |
|--------|---------|
| `offline_compute_features.py --oid <bigint> [--credentials PATH]` | DB → message → features for one oid; prints a populated long frame. |
| `offline_benchmark_features.py [--n N] [--warmup K] [--min-det M] [--credentials PATH]` | Times `compute_features` over N real oids (extractor built once, warm-up excluded). |
| `offline_compare_vs_alerce.py --oid <bigint|ZTFstr> [--version V] [--rtol] [--atol]` | Runs our pipeline on a multisurvey oid, maps it to its ZTF string oid via `idmapper`, fetches `alerce.feature` for that object, and diffs. |

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

## Fixes this relies on

The ZTF feature path in `features.utils.parsers` and `lc_classifier` had to be made
consistent with the `correction-ztf` message contract (these are committed on the
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

**Status:** the pipeline produces correct, populated features on real objects.
A true value-level equality check against `alerce.feature` is pending a
multisurvey backfill — the multisurvey DB is currently a recent ~3-month slice,
so its light curves are far shorter than the legacy full-history ones, and a
comparison today reflects the input-LC difference rather than the computation.
Pass `--version` matching the deployed `lc_classifier` for an apples-to-apples
comparison once the data is backfilled.
