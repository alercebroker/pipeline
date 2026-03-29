# ALeRCE Pipeline Monorepo

This is the ALeRCE broker pipeline monorepo. It contains multiple pipeline steps that process astronomical alerts from 
streams like ZTF and LSST. It has shared libraries, ML classifiers, etc., all written in Python.

## Monorepo Structure

Each top-level `*_step` directory is an independent pipeline step (microservice):
- `ingestion_step/` – Ingests raw alerts from ZTF/LSST streams
- `sorting_hat_step/` – Routes alerts to appropriate streams
- `prv_candidates_step/` – Processes previous candidates
- `correction_step/` / `correction_multisurvey_step/` – Photometric corrections
- `lightcurve-step/` – Builds light curves
- `magstats_step/` – Computes magnitude statistics
- `xmatch_step/` – Cross-matches with external catalogs
- `feature_step/` – Extracts features from light curves
- `stamp_classifier_step/` / `stamp_classifier_2025_step/` – Stamp-based classifiers
- `early_classification_step/` – Early classification
- `lc_classification_step/` – Light curve classification
- `lc_anomaly_step/` – Anomaly detection
- `metadata_step/` – Handles alert metadata
- `alert_archiving_step/` – Archives alerts to S3
- `watchlist_step/` – Watchlist matching
- `reflector_step/` – Reflector stream step
- `s3_step/` / `s3_multisurvey_step/` – S3 storage steps
- `scribe/` / `scribe_multisurvey/` – Database writer steps. Many steps send messages to scribe via Kafka, and scribe handles writing to the database.
- `magstats_multisurvey_step/` – Multisurvey magnitude statistics
- `rubin_stamp_classifier_step/` – Stamp-based classifier for Rubin/LSST alerts

Shared components:
- `libs/` – Shared Python libraries used across steps:
  - `apf/` – Alert Processing Framework (APF): the base framework all steps are built on
  - `survey_parser_plugins/` – Normalizes alerts from different surveys into a generic schema
  - `db-plugins/` / `db-plugins-multisurvey/` – Database abstraction plugins
  - `idmapper/` – Converts catalog object IDs (ZTF, LSST) to ALeRCE master IDs
  - `lsst_schema_parser/` – Parses LSST alert schemas
  - `alerts_store/` – Alert storage utilities
  - `xmatch_client/` – Client for cross-matching with external catalogs
  - `test_utils/` – Common utilities and mock data for testing steps
- `schemas/` – Avro/JSON schemas for Kafka messages between steps
- `lc_classifier/` – Core light curve classifier library
- `alerce_classifiers/` – (git submodule at repo root) Classifier model weights and configs
- `P4J/` – Period-finding library used by feature_step
- `mhps/` – MHPS feature calculation used by feature_step
- `charts/` – Helm charts for Kubernetes deployment
- `ci/` – CI/CD scripts (builds use Dagger via `ci/build.py`)
- `training/` – Model training notebooks and scripts
- `_utils/` – Shared utility scripts

## Each Pipeline Step Follows This Pattern
- `<step_name>/` 
  - `pyproject.toml` – Defines the Python package for the step
  - `Dockerfile` – Container definition
  - `<step_name>/` – Main Python package
  - `scripts/` – Contains the main entrypoint script and auxiliary scripts
  - `tests/` – Unit and integration tests
  - `README.md` – Documentation for the step

## Key Conventions
- Each step is an independent Python package managed with **Poetry** (`pyproject.toml`).
- All steps are built on the **APF (Alert Processing Framework)** (`libs/apf/`), which provides the base step class, Kafka consumer/producer wiring, and deployment scaffolding.
- Steps communicate via **Apache Kafka** using Avro schemas defined in `schemas/`.
- Each step runs as a Docker container in Kubernetes, with Helm charts in `charts/`.
- Shared libraries live in `libs/` and are installed as local path dependencies during Docker builds.
- Tests use **pytest** and are run in CI on GitHub Actions; CI builds use **Dagger**.
- Many steps have `multisurvey` variants (suffix `_multisurvey_step` or prefix `_ms_`) that support multiple astronomical surveys (ZTF, LSST/Rubin, ATLAS) in contrast to older ZTF-only steps. We are migrating towards multisurvey support across all steps, and will eventually deprecate the older single-survey steps.