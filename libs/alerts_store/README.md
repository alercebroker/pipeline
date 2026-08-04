# alerts_store

Consumes an LSST alert Kafka topic and stores each alert as an `.avro` file on disk
(`schemaless_writer`, one file per alert under `<data-dir>/<mjd>/`), plus an optional
web + file-server layer to browse/search the stored alerts.

Config lives in `config.yaml` (gitignored — carries the Kafka SASL password); copy
`config.yaml.example` and fill it in. Deployment-specific paths/binds come from a
gitignored `.env` (see `.env.example`).

## Running modes

The compose stack is two independent halves:

- **Writer** — `alerts_store`: consumes the Kafka topic and writes `.avro` files to the
  data dir, maintaining `index.txt` for new alerts. Opens no port.
- **Serve layer (optional)** — `web` (FastAPI search UI) + `file_server` (nginx). Only
  `file_server` exposes anything over HTTP.

**Store only, no exposure:**

    docker compose up -d alerts_store
    docker compose up -d alerts_store indexer_v11_1   # + backfill/safety re-index

**Take exposure down on a running stack:**

    docker compose stop web file_server

Nothing serves data over HTTP unless `file_server` is running.
