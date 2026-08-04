# An in-depth guide to the ALeRCE pipeline

Every step is an event-driven microservice that listens to one or more Kafka topics, performs its operation, and produces to a downstream topic and/or writes to a database. Steps are built on the [APF framework](https://github.com/alercebroker/APF), which dictates the step lifecycle (`pre_execute` → `execute` → `post_execute` → `produce`) and provides the (de)serialization hooks.

This document describes both pipelines that live in this repo: the **legacy single-survey pipeline** (running in production) and the **multisurvey pipeline** (in staging on the `multisurvey` branch).

The diagram below describes the legacy single-survey topology and is currently outdated relative to the multisurvey rewrite:

![Legacy pipeline diagram](https://user-images.githubusercontent.com/20263599/229163793-f0cefe89-6a2b-4dee-a111-20da2eec3461.png)

> **TODO:** replace with an up-to-date multisurvey topology diagram.

---

## Multisurvey pipeline

The multisurvey pipeline replaces the legacy "one codebase per survey" approach with a single image per step, parameterised by a survey strategy. Survey-tagged Kafka topics carry messages through the pipeline. Object identifiers are unified as 64-bit integers via the [idmapper](./libs/idmapper) library.

**Deployment pattern.** Multisurvey steps share a single generic Helm chart, [`charts/multisurvey_step/`](./charts/multisurvey_step/). Each deployment supplies its own values overlay that sets the container image, namespace, and `configYaml` (Kafka consumer/producer topics, schemas, PSQL connection, survey strategy, …). For example, ZTF ingestion in staging is deployed via `charts/multisurvey_step/values.ztf.staging.yaml` with `image.repository: ghcr.io/alercebroker/ingestion_step`. The same chart is reused for correction, magstats, scribe, s3, and the stamp classifiers — each with its own image and overlay. (The legacy `charts/correction_multistream_ztf_step/` is a separate one-off chart kept alongside this pattern.)

**ZTF flow (confirmed deployed in staging):**

```
[external ZTF broker]
      │
      ▼
reflector_step                        topic in: ztf
                                      topic out: ztf-reflector
      │
      ▼
ingestion_step (SURVEY=ZTF)           topic in:  ztf-reflector
      │                               topic out: ztf-ingestion
      │  also writes objects + detections + forced-phot + non-det → PSQL
      ▼
correction_multisurvey_step (ZTF)     topic in:  ztf-ingestion
      │                               topic out: correction-ms-ztf
      │  also emits scribe commands → scribe-multisurvey
      ▼
magstats_multisurvey_step (ZTF)       topic in:  correction-ms-ztf
                                      topic out: TODO (chart absent)
                                      scribe   → scribe-multisurvey

stamp_classifier_2025_multisurvey_step (ZTF, parallel)
                                      topic in: TODO (likely ztf-reflector or raw)
                                      writes probabilities → PSQL + scribe-multisurvey

s3_multisurvey_step (parallel)        topic in: TODO (chart absent)
                                      writes → S3 (per-survey bucket)

scribe_multisurvey                    topic in: scribe-multisurvey
                                      writes → PSQL (multisurvey schema)
```

**LSST flow (deployed in production at the `db1` cluster):**

```
[external LSST broker @ usdf-alert-stream-dev.lsst.cloud:9094]
      │  topic: lsst-alerts-v10.0  (Avro v10.0)
      ▼
reflector_step                        ns: multisurvey-reflector-lsst-db1-v10-step
      │                               topic in:  lsst-alerts-v10.0 (external)
      │                               topic out: lsst (internal)
      ▼
[internal broker @ 192.168.50.106:29092]
      │  topic: lsst (Avro v10.0 — pass-through from reflector)
      │
      ├──► ingestion_step (SURVEY=LSST)         ns: multisurvey-ingestion-lsst-db1-step
      │       topic in:  lsst                   topic out: lsst-ingestion
      │       also writes objects + detections + forced-phot → PSQL
      │       │
      │       ▼
      │    correction_multisurvey_step (LSST)   ns: multisurvey-correction-lsst-db1-step
      │       topic in:  lsst-ingestion         topic out: lsst-correction
      │       scribe → scribe-multisurvey
      │       │
      │       ▼
      │    magstats_multisurvey_step (LSST)     ns: multisurvey-magstats-lsst-db1-step
      │       topic in:  lsst-correction        topic out: lsst-magstats
      │       scribe → scribe-multisurvey
      │       │
      │       ▼
      │    rubin_features (LSST)                ns: multisurvey-rubin-features-db1-step
      │       topic in:  lsst-magstats          (terminal — no Kafka output)
      │       computes features (with inline xmatch) → PSQL + scribe-multisurvey
      │
      └──► rubin_stamp_classifier_step (parallel, two ML-model variants)
              ns: multisurvey-rubin-stamp-classifier-db1-v10-step
              ns: multisurvey-rubin-stamp-classifier-db1-20260409-step
              topic in: lsst (Avro v10.0)       topic out: rubin_stamp_classifier
              writes probabilities → PSQL + scribe-multisurvey
```

> The LSST overlays (reflector, ingestion, correction, magstats, rubin_features,
> stamp classifiers) are not committed under [`charts/multisurvey_step/`](./charts/multisurvey_step/);
> they are applied via uncommitted overlays at the cluster. Schema version on the
> wire today is **Rubin v10.0**; a planned hard-cut to v11 is tracked in
> [lsst_v11_plan.md](./lsst_v11_plan.md).

### reflector_step

Custom Kafka mirror — copies one or more topics from an external cluster into the internal cluster without (de)serialization. No DB connection. See [reflector_step/README.md](./reflector_step/README.md) for the full deploy config.

### ingestion_step

The multisurvey gateway. Replaces `sorting_hat_step` + `prv_candidates_step` for multisurvey. It:

- Parses raw alerts using a survey-specific strategy (ZTF or LSST).
- Applies survey-specific transforms (`jd_to_mjd`, `fid_to_band`, `forcediffimflux_to_mag`, `isdiffpos_to_int`, …; full list in [ingestion_step/ingestion_step](./ingestion_step)/`*/transforms.py`).
- Resolves the unified 64-bit `oid` for each object via [idmapper](./libs/idmapper).
- Inserts objects, detections, forced photometry, and non-detections directly into the multisurvey PSQL schema (no scribe hop).
- Produces a normalised, survey-tagged message to the next topic.

For ZTF, this includes decoding the `prv_candidates` binary in `extra_fields` (handled by `ZtfPrvCandidatesExtractor`).

**Schemas:** see [schemas/ingestion_step/](./schemas/ingestion_step/).

### correction_multisurvey_step

Folds the role of the legacy `lightcurve-step` and `correction_step` into one operation:

- Queries the multisurvey PSQL schema for the full historical lightcurve of each incoming object.
- Joins it with the current batch.
- Applies the difference-to-apparent magnitude correction (survey-aware).
- Emits scribe commands to `scribe-multisurvey` for derived writes.

Per-survey output schemas live under [schemas/correction_ms_step/](./schemas/correction_ms_step/) (e.g. `lsst/output.avsc`).

### magstats_multisurvey_step

Computes per-band magnitude statistics for each object: `ndet`, `firstmjd`, `lastmjd`, mean coordinates, etc. Computes object-level aggregates. Emits scribe commands (`magstat`, `magstat_objects`) on `scribe-multisurvey`.

LSST deployment (`multisurvey-magstats-lsst-db1-step`) consumes `lsst-correction`
and produces to `lsst-magstats`. ZTF overlay still pending — confirm once committed.

### rubin_features

LSST features step. Consumes `lsst-magstats` (reader schema:
[schemas/magstats_ms_step/lsst/output.avsc](./schemas/magstats_ms_step/lsst/output.avsc)),
computes lightcurve features, and writes them via scribe to PSQL. **Terminal step** —
no `PRODUCER_CONFIG`; no Kafka output.

Notable architectural difference from the legacy [feature_step](./feature_step):
xmatch is folded **inline** (controlled by `USE_XMATCH: true` and `XMATCH_CONFIG.base_url`)
rather than being a separate Kafka hop. The legacy pipeline runs `xmatch_step` as a
distinct service; multisurvey calls the xmatch service from inside the features step.

LSST deployment: `multisurvey-rubin-features-db1-step`.

### stamp_classifier_2025_multisurvey_step

ZTF stamp classifier targeting the multisurvey schema. Reads the 63×63 FITS cutouts (`cutoutScience`, `cutoutTemplate`, `cutoutDifference`) directly from the alert. Classes: SN / AGN / VS / asteroid / bogus / satellite. Writes probabilities to PSQL and emits a scribe command.

**Output schema:** [schemas/stamp_classifier_2025_multisurvey_step/output.avsc](./schemas/stamp_classifier_2025_multisurvey_step/output.avsc) — fields: `oid`, `sid`, `measurement_id`, `probabilities`.

> **TODO:** no values overlay for this step is committed yet, so the input topic is not pinned. Likely `ztf-reflector` since the step reads raw alert fields — confirm once an overlay lands.

This is the multisurvey-schema successor to `stamp_classifier_2025_step`. The plain version targets the legacy `db-plugins` schema and emits a string `objectId`; the `_multisurvey` variant uses [idmapper](./libs/idmapper) to map to the numeric `oid` and writes via `db-plugins-multisurvey`.

### rubin_stamp_classifier_step

LSST counterpart to the ZTF stamp classifier. Consumes the raw `lsst` topic (Avro
schema **v10.0** in production today; v11 planned — see
[lsst_v11_plan.md](./lsst_v11_plan.md)). Splits non-solar-system vs solar-system
objects (writing `diaObjectId` or `ssObjectId` accordingly). Probabilities are
written to PSQL and to `scribe-multisurvey`.

**Output topic:** `rubin_stamp_classifier` (schema: `RubinStampClassifierOutput`).

**Two deployments run in parallel**, same image, different ML model configurations
(distinct `MODEL_CONFIG.MODEL_PATH`, `CLS_ID`, etc.):

- `multisurvey-rubin-stamp-classifier-db1-v10-step`
- `multisurvey-rubin-stamp-classifier-db1-20260409-step`

Each is its own consumer group on the `lsst` topic, so both see every alert
independently and write classifications under their own `classifier_id`.

Note: the source code at [rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py](./rubin_stamp_classifier_step/rubin_stamp_classifier_step/step.py)
contains v7.4-era hardcoded defaults for fields that were absent in v7.4 (`airmass`,
`magLim`, `seeing` — still absent in v11; `scienceFlux`, `scienceFluxErr` — present
since v10). The latter two appear to be unconditionally overwritten with `0.0`,
which is suspected silent data loss; tracked in [lsst_v11_plan.md](./lsst_v11_plan.md).

### s3_multisurvey_step

Uploads raw Avro alerts to per-survey S3 buckets in parallel using a thread pool. Survey is selected by `SURVEY_ID`, bucket by `BUCKET_CONFIG`.

> **TODO:** no values overlay for this step is committed yet, so the consumer topic is not pinned (raw survey topic vs. post-ingestion topic). Confirm once an overlay lands.

### scribe_multisurvey

CQRS async DB writer for the multisurvey PSQL schema. Consumes JSON command messages from the `scribe-multisurvey` Kafka topic and executes bulk SQL upserts. Each message carries a `step` discriminator (e.g. `"magstat"`, `"magstat_objects"`, `"probability-archival-step"`) that determines which table is written. Deduplicates `magstat` / `magstat_objects` messages within a batch.

> **TODO:** no values overlay for `scribe_multisurvey` is committed yet under `charts/multisurvey_step/`. The legacy `charts/scribe/` deploys the MongoDB scribe; the multisurvey scribe is expected to deploy via the generic `multisurvey_step` chart with its own image + overlay.

### Legacy steps with no multisurvey equivalent

The following legacy steps do not yet have multisurvey counterparts. The legacy single-survey pipeline continues to fill these roles for ZTF in production.

- `lc_classification_step`
- `watchlist_step`
- `early_classification_step`

`feature_step` and `xmatch_step` have an LSST-side multisurvey equivalent in
[`rubin_features`](#rubin_features) (xmatch folded inline). A ZTF features step on
the multisurvey schema does not yet exist — `rubin_features` is, by name and as
deployed, LSST-only.

> **TODO:** clarify roadmap — are the remaining legacy steps planned for multisurvey, or deliberately staying on the legacy path? Will the LSST `rubin_features` design (xmatch inline) be generalised to ZTF, or will ZTF get a separate multisurvey features step?

---

## Legacy pipeline (single-survey)

This is the topology currently running in production (confirmed via `ci/helm_values/production/main.tf`). One deployment per survey (ZTF, ATLAS).

### Sorting Hat

Gateway step: assigns an ALeRCE ID (`aid`) to every incoming alert. The `aid` is a string starting with `AL` followed by the last two digits of the current year, encoding the truncated alert position. The step queries the object database to decide whether the alert reuses an existing `aid` (matched by `oid` or by 1.5″ conesearch) or generates a new one.

One instance per survey, sharing config except for the input/output topics.

**Output schema:** [schemas/sorting_hat_step/output.avsc](./schemas/sorting_hat_step/output.avsc).

### Previous Candidates

Decodes the `prv_candidates` binary inside `extra_fields` for ZTF alerts and produces a list of objects, each with their `aid`, detections (the main alert plus any previous detections), and non-detections. For ATLAS, only the main alert is passed and the non-detections list is empty.

**Output schema:** [schemas/prv_candidate_step/output.avsc](./schemas/prv_candidate_step/output.avsc).

### Lightcurve

Retrieves the full stored lightcurve (detections + non-detections) for each object from the database and merges it with the current batch. Repeated objects within the same batch are merged.

**Output schema:** same as the previous-candidates step. See [schemas/lightcurve_step/output.avsc](./schemas/lightcurve_step/output.avsc).

### Correction

Applies the difference-to-apparent magnitude correction on each detection. Also computes the object's mean RA/Dec from its detections.

**Output schema:** [schemas/correction_step/output.avsc](./schemas/correction_step/output.avsc).

### Magstats

Generates / updates the object record that is stored in the DB.

**Output schema:** [schemas/magstats_step/output.avsc](./schemas/magstats_step/output.avsc).

### Xmatch

Crossmatch against the AllWISE catalog via the CDS crossmatch service. Sends results to the `xmatch` topic.

**Output schema:** [schemas/xmatch_step/output.avsc](./schemas/xmatch_step/output.avsc).

### Features

Computes lightcurve features used by downstream classifiers.

**Output schema:** [schemas/feature_step/output.avsc](./schemas/feature_step/output.avsc).

### LC Classifier

Hierarchical classifiers (Balto, Messi, Toretto, …) run on the features. May skip an object if required features are missing. Produces to a daily-stamped topic (`lc_classifier_YYYYMMDD`).

### Stamp Classifier (legacy)

ML model that classifies alerts based on their stamps. One instance per survey (ZTF, ATLAS). Writes results via scribe to MongoDB; does not produce to a downstream Kafka topic.

**Output schema:** [schemas/stamp_classifier_step/output.avsc](./schemas/stamp_classifier_step/output.avsc) (note: the directory is `stam_classifier_step` in older paths — verify the on-disk name when consuming).

### Other legacy steps

- **early_classification_step** — Pre-feature classifier. **TODO:** input/output topics not enumerated here; check chart values.
- **metadata_step** — ZTF-specific. Consumes `sorting-hat` (production consumes `sorting-hat`; the chart default `values.yaml` still lists `sorting-hat-ztf`), parses ZTF `extra_fields` (PS1, Gaia cross-match data), and writes the crossmatch metadata tables (`reference`, `ss_ztf`, `ps1_ztf`, `dataquality`, `gaia_ztf`) **directly** to PSQL via on-conflict upsert — not through scribe. No Kafka data output (metrics only).
- **watchlist_step** — Crossmatches alerts against user-defined watchlists; notifies users on hits.
- **alert_archiving_step** — Archives raw Avro alerts to S3 in daily-partitioned paths (`avro_YYYYMMDD/`, snappy codec). Configured for ZTF and ATLAS buckets.
- **s3_step** — Uploads Avro files to an AWS S3 bucket (legacy archival).
- **scribe** — CQRS async writer for MongoDB. Consumes commands from the `w_object` topic.
- **lc_anomaly_step** — Anomaly-detection variant of the LC classifier. Shares the same `LateClassifier` class as `lc_classification_step`, but configured with a different model class. **TODO:** README is empty and no chart was found; deployment status unclear.

---

## Legacy ↔ multisurvey differences

| Concern | Legacy | Multisurvey |
|---|---|---|
| Object ID | String `aid` (`AL25…`) + survey `oid` (`ZTF21abcdefg`) | Numeric 64-bit `oid` (bit-packed; resolved by `idmapper`) |
| DB schema | MongoDB (objects, probabilities, features) + PSQL (users, …) | Unified PSQL multisurvey schema (`db-plugins-multisurvey`), hash-partitioned |
| Gateway | `sorting_hat_step` (per survey) | `ingestion_step` (one image, survey strategy) |
| Lightcurve retrieval | Dedicated `lightcurve-step` | Folded into `correction_multisurvey_step` |
| Scribe | MongoDB (`scribe`, topic `w_object`) | SQL (`scribe_multisurvey`, topic `scribe-multisurvey`) |
| Surveys supported | ZTF + ATLAS | ZTF + LSST (ATLAS not present in multisurvey code) |
| ZTF prv_candidates decoding | Dedicated `prv_candidates_step` | Inside `ingestion_step` (`ZtfPrvCandidatesExtractor`) |

> **TODO:** the multisurvey pipeline does not yet cover xmatch / features / lc_classification / watchlist / early-classification. Until those are ported, the legacy single-survey pipeline remains the production path for ZTF.
