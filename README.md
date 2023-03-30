# Lightcurve step

This step connects to the database and retrieves the full lightcurve for the given ALeRCE objects.

If one or more of the incoming messages belong to the same object, they will be merged together, always keeping
detections with stamps over those without. This circumstance can arise from a newer detection coming in listing
another one as a previous detection (thus without stamp), even though the previous detection was received in the 
same stream.

Non-detections are also merged, keeping a single copy of any non-detection that shares the same `mjd`, `oid` and
`fid`.

If a detection or non-detection already appears in the database, the one taken from it will be kept
over any that comes with the incoming messages.

#### Previous step:
- [Previous candidates](https://github.com/alercebroker/prv_candidates_step)

#### Next step:
- [Correction](https://github.com/alercebroker/correction_step)

## Environment variables

These are required only when running the scripts, as is the case for the Docker images.

### Consumer setup

- `CONSUMER_SERVER`: Kafka host with port, e.g., `localhost:9092`
- `CONSUMER_TOPICS`: Some topics. String separated by commas, e.g., `topic_one` or `topic_two,topic_three`
- `CONSUMER_GROUP_ID`: Name for consumer group, e.g., `correction`
- `CONSUME_TIMEOUT`: (optional) Timeout for consumer
- `CONSUME_MESSAGES`: (optional) Number of messages consumed in a batch

### Producer setup

- `PRODUCER_SERVER`: Kafka host with port, e.g., `localhost:9092`
- `PRODUCER_TOPIC`: Topic to write into for the next step

[//]: # (### SSL authentication)

[//]: # ()
[//]: # (When using SSL authentication for the whole cluster, the following must be provided)

[//]: # ()
[//]: # (- `KAFKA_USERNAME`: Username for the step authentication)

[//]: # (- `KAFKA_PASSWORD`: Password for the step authentication)

### Metrics producer setup

- `METRICS_TOPIC`: (optional) Topic name, e.g., `topic_one`
- `METRICS_SERVER`: Kafka host with port, e.g., `localhost:9092`

## Run the released image

For each release, an image is uploaded to GitHub packages. To download:

```bash
docker pull ghcr.io/alercebroker/lightcurve_step:latest
```
## Local Installation

### Requirements

To install the repository specific packages run:
```bash
pip install -r requirements.txt
```

### Development

The following additional packages are required to run the tests:
```bash
pip install pytest pytest-docker
```

Run tests using:
```bash
python -m pytest tests
```

