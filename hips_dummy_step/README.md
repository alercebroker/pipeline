# HiPS dummy step

A throughput test. For every alert it consumes, the step sends the alert's
position to an image service and receives catalog reference images of that
region of the sky. The images are loaded and dropped: nothing is produced to
Kafka or written to a database. The consumer group's lag shows whether the
step and the service keep up with the alert stream.

Later these images are meant as an input for ML models.

The same image runs as two instances with different configuration: one for
ZTF and one for LSST, each with its own consumer group.

## How it works

1. `pre_execute` reads each alert's position (`candidate.ra/dec` for ZTF,
   `diaSource.ra/dec` for LSST). It wraps ra into [0, 360) and drops alerts
   whose position is missing, not finite, or has dec outside [-90, 90], since
   one bad position would make the service reject the whole request. Dropped
   alerts are logged and counted.
2. `execute` splits the batch into requests of at most
   `MAX_POSITIONS_PER_REQUEST` positions and sends them one after another.
   Positions the service can't serve come back in its `errors` list; they are
   counted, not retried.
3. The consumer offset is committed once every request in the batch has been
   answered.

**Busy service.** When the service answers HTTP 429, the client waits for the
`Retry-After` time (2 s if missing) and retries. Server errors (5xx), timeouts
and connection errors are retried with exponential backoff (1, 2, 4… s, capped
at `MAX_BACKOFF`). No alert is dropped because the service is busy or down: the
step waits, and the lag grows. Any other HTTP error (e.g. 400) means the request
is wrong, so the step stops with an error.

## Configuration

Read from the YAML file at `CONFIG_YAML_PATH` (default `/config/config.yaml`).

```yaml
SURVEY: ztf                     # ztf | lsst: where to read the position from
CONSUMER_CONFIG:
  CLASS: apf.consumers.KafkaConsumer   # LSST: apf.consumers.KafkaSchemalessConsumer + SCHEMA_PATH
  TOPICS: ["<alert topic>"]
  PARAMS:
    bootstrap.servers: "<brokers>"
    group.id: "<new consumer group>"
    auto.offset.reset: latest
  consume.messages: 100
  consume.timeout: 10
IMAGE_SERVICE_CONFIG:
  URL: "<full endpoint URL>"     # or the IMAGE_SERVICE_URL environment variable
  BANDS: ["r"]                   # bands to request
  HEADERS: false                 # also request the images' WCS headers
  REFERENCE_SURVEY: null         # optional: ask for one specific catalog
  MAX_POSITIONS_PER_REQUEST: 200
  TIMEOUT: 120                   # seconds per request
  MAX_BACKOFF: 60                # seconds, cap for the retry backoff
FEATURE_FLAGS:
  PROMETHEUS: true               # serve metrics on port 8000
LOGGING_DEBUG: false
```

The service URL and the Kafka credentials are not in this repository. When
deployed they come from the release's Helm values; locally, from an untracked
`config.yaml` (git-ignored) or a `.env` file setting `IMAGE_SERVICE_URL`.

There is no `PRODUCER_CONFIG` or `METRICS_CONFIG`: the step produces nothing.

## Metrics

On top of the framework's step metrics (`messages_consumed`, `batch_processing_seconds`, …):

| Metric | Meaning |
| --- | --- |
| `image_service_request_duration_seconds` | Time per successful request, including download and loading the images |
| `image_service_positions_requested_total` | Positions sent and answered |
| `image_service_positions_failed_total` | Positions answered without images |
| `image_service_positions_skipped_total` | Alerts not sent because of an invalid position |
| `image_service_response_bytes_total` | Size of the responses, after decompression |
| `image_service_retries_total{reason}` | Retries: `busy`, `server_error`, `connection` |

## Development

The framework pins `fastavro==1.6.1`, which only has prebuilt packages up to
Python 3.10, so use a 3.10 interpreter:

```bash
poetry env use /path/to/python3.10
poetry install --with test
poetry run pytest tests
```

Run locally against a reachable service and Kafka:

```bash
CONFIG_YAML_PATH=config.yaml poetry run python scripts/run_step.py
```

Build the image from the repository root:

```bash
docker build -f hips_dummy_step/Dockerfile -t hips_dummy_step .
```
