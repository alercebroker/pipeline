# Pipeline

This monorepo contains the ALeRCE alert-processing pipeline. Each step is an event-driven microservice that consumes from a Kafka topic, performs its operation, and produces to the next topic (or writes to a database). Steps are built on the [APF framework](https://github.com/alercebroker/APF).

Two pipelines coexist in this repo:

- **Legacy single-survey pipeline** — the topology currently running in production. One deployment per survey (ZTF, ATLAS).
- **Multisurvey pipeline** — the survey-agnostic rewrite (ZTF + LSST) currently being rolled out in staging. Lives alongside the legacy code on this branch.

See [PIPELINE.md](./PIPELINE.md) for in-depth descriptions of each step, their input/output schemas, and known gaps.

## Multisurvey pipeline (staging)

Survey-agnostic redesign. One step image is reused across surveys via a strategy pattern, configured at deploy time (`SURVEY_STRATEGY` / `SURVEY`).

```
[external survey broker]
        │
        ▼
  reflector_step                    (raw Kafka mirror — no deserialization)
        │
        ▼
  ingestion_step                    (per-survey strategy: ZTF / LSST)
        │   inserts objects, detections, forced-phot, non-detections into PSQL
        ▼
  correction_multisurvey_step       (retrieves historical lightcurve, applies
        │                            mag correction; emits scribe commands)
        ▼
  magstats_multisurvey_step         (computes magstats; emits scribe commands)
        │
        ▼
  rubin_features                    (LSST only: features + inline xmatch;
                                     terminal — writes via scribe + PSQL)

  stamp classifiers (run in parallel, consume raw / ingestion topics):
    - stamp_classifier_2025_multisurvey_step   (ZTF)
    - rubin_stamp_classifier_step              (LSST, two model variants)

  Sinks:
    - scribe_multisurvey            (CQRS writer: bulk SQL upserts to PSQL)
    - s3_multisurvey_step           (archives raw Avro to S3 per survey)
```

### Multisurvey steps

- [ingestion_step](./ingestion_step) — Gateway. Parses raw alerts, applies survey-specific transforms (e.g. `jd_to_mjd`, `fid_to_band`), and writes objects/detections to the multisurvey PSQL schema. Replaces `sorting_hat_step` + `prv_candidates_step` for multisurvey.
- [correction_multisurvey_step](./correction_multisurvey_step) — Retrieves the full historical lightcurve from the DB, joins it with the incoming alert, and applies the difference-to-apparent magnitude correction. Folds in the role of the legacy `lightcurve-step`.
- [magstats_multisurvey_step](./magstats_multisurvey_step) — Computes per-band magnitude statistics (ndet, firstmjd, lastmjd, mean coordinates) and object-level stats.
- [scribe_multisurvey](./scribe_multisurvey) — CQRS async writer. Consumes commands from the `scribe-multisurvey` topic and executes bulk SQL upserts.
- [s3_multisurvey_step](./s3_multisurvey_step) — Uploads raw Avro alerts to per-survey S3 buckets in parallel.
- [stamp_classifier_2025_multisurvey_step](./stamp_classifier_2025_multisurvey_step) — ZTF stamp classifier targeting the multisurvey PSQL schema. Classifies into SN / AGN / VS / asteroid / bogus / satellite.
- [rubin_stamp_classifier_step](./rubin_stamp_classifier_step) — LSST stamp classifier. Splits non-solar-system vs solar-system objects, writes probabilities to PSQL and to `scribe-multisurvey`. Two deployments run in parallel with different ML models.
- `rubin_features` (deployed as `multisurvey-rubin-features-db1-step`) — LSST features step. Consumes `lsst-magstats`, computes lightcurve features with **inline xmatch** (no separate xmatch hop), and writes via scribe + PSQL. Terminal step — no Kafka output.
- [reflector_step](./reflector_step) — Custom Kafka mirror (lightweight MirrorMaker replacement). Used to replicate external survey topics into the internal cluster.

> **TODO:** `lc_classification`, `watchlist`, and `early_classification` do **not** yet have multisurvey equivalents — the legacy steps continue to fill these roles for ZTF in production. `feature_step` and `xmatch_step` have an LSST counterpart in `rubin_features` (xmatch folded inline), but no ZTF-side multisurvey features step exists yet.

## Legacy pipeline (production)

```
  sorting_hat_step  →  prv_candidates_step  →  lightcurve-step  →  correction_step
                                                                         │
                              ┌──────────────────────────────────────────┤
                              ▼                                          ▼
                       magstats_step                              xmatch_step
                                                                         │
                                                                         ▼
                                                                  feature_step
                                                                         │
                                                                         ▼
                                                              lc_classification_step

  Parallel branches off sorting-hat / earlier topics:
    - stamp_classifier_step       (per survey: ZTF, ATLAS)
    - early_classification_step   (pre-feature classifier)
    - metadata_step               (ZTF: PS1/Gaia metadata write)
    - watchlist_step              (user watchlist crossmatch + notify)
    - alert_archiving_step        (raw Avro → S3, daily partitions)
    - s3_step                     (legacy Avro upload)
    - scribe                      (CQRS writer: MongoDB)
```

### Legacy steps

- [sorting_hat_step](./sorting_hat_step) — Assigns an ALeRCE ID (`aid`) to incoming alerts. One instance per survey.
- [prv_candidates_step](./prv_candidates_step) — Decodes the ZTF `prv_candidates` binary in `extra_fields` and yields the previous detections / non-detections.
- [lightcurve-step](./lightcurve-step) — Retrieves the full stored lightcurve for each object and merges it with the new batch.
- [correction_step](./correction_step) — Applies difference-to-apparent magnitude correction; computes mean RA/Dec.
- [magstats_step](./magstats_step) — Generates / updates the object record stored in the DB.
- [xmatch_step](./xmatch_step) — Crossmatch with AllWISE via the CDS service.
- [feature_step](./feature_step) — Computes lightcurve features used by classifiers.
- [lc_classification_step](./lc_classification_step) — Hierarchical classifiers run on the features (Balto, Messi, Toretto, …).
- [stamp_classifier_step](./stamp_classifier_step) — Stamp-based classifier; one instance per survey.
- [early_classification_step](./early_classification_step) — Pre-feature classifier.
- [metadata_step](./metadata_step) — ZTF-specific: parses ZTF `extra_fields` and writes PS1 / Gaia cross-match metadata to PSQL.
- [watchlist_step](./watchlist_step) — Crossmatch against user-defined watchlists; notifies users on hits.
- [alert_archiving_step](./alert_archiving_step) — Archives raw Avro alerts to S3 in daily-partitioned files (ZTF + ATLAS).
- [s3_step](./s3_step) — Uploads Avro files to an AWS S3 bucket.
- [scribe](./scribe) — CQRS async writer for MongoDB.
- [lc_anomaly_step](./lc_anomaly_step) — Anomaly-detection variant of the LC classifier (shares the `LateClassifier` class with `lc_classification_step`, configured with a different model). **TODO:** confirm deployment status.

## Glossary

### Concepts shared across both pipelines

- **Alert** — Incoming detection from a survey.
- **Survey** — Observational project that generates alerts (may use one or more telescopes).
- **Object** — A spatially clustered set of detections assumed to come from the same source.
- **Detection** — Alert stored in the database for which a significant flux change vs. its template image was measured.
- **Non-detection** — Stream observation in which no significant flux change was detected. Not delivered as alerts; in ZTF they arrive bundled inside an alert's previous candidates.

### Multisurvey-specific concepts

- **`oid` (Object ID)** — In multisurvey, a **64-bit integer** (not a string like `ZTF21abcdefg`). High bits encode a survey prefix, low bits encode the survey-specific id; computed by the [idmapper](./libs/idmapper) library. Replaces both the legacy string `oid` and the legacy `aid` (`AL25...`).
- **`sid` (Survey ID)** — Small integer identifying the alert stream sub-type. `0` = ZTF, `1` = LSST DIA Object, `2` = LSST SS Object. Most multisurvey tables use `(oid, sid)` as the composite primary key.
- **`tid` (Telescope ID)** — Small integer identifying the physical telescope. `0` = ZTF, `1` = LSST/Rubin.
- **`measurement_id`** — Survey-specific per-detection identifier. ZTF: `candid`. LSST: `diaSourceId`.
- **DIA Object / DIA Source** — LSST terminology. A *DIA Object* is the sky-position record grouping related difference-image alerts; a *DIA Source* is one such measurement. LSST also has *SS Object* / *SS Source* for solar-system bodies.
- **Forced photometry** — Flux measurement at a known position even when no source is detected above threshold. ZTF: `fp_hists`. LSST: `forced_sources`.
- **Survey strategy** — Design pattern used in `ingestion_step` and `correction_multisurvey_step`. One image, one process per survey; the survey-specific parser/extractor is selected at startup from a `SURVEY_STRATEGY` / `SURVEY` config key.
