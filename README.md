# Magnitude correction step

## Description

Process alerts and applies correction to pass from difference magnitude to apparent magnitude. This
includes the previous detections coming from the ZTF alerts.

Presently, there is no correction for ATLAS alerts.

#### Previous step:
- [Previous candidates](https://github.com/alercebroker/prv_candidates_step)

#### Next step:
- [Light curve](https://github.com/alercebroker/lightcurve-step)

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

### Scribe producer setup

The [scribe](https://github.com/alercebroker/alerce-scribe) will write results in the database. 

- `SCRIBE_TOPIC`: Topic name, e.g., `topic_one`
- `SCRIBE_SERVER`: Kafka host with port, e.g., `localhost:9092`

### Metrics producer setup

- `METRICS_TOPIC`: (optional) Topic name, e.g., `topic_one`
- `METRICS_SERVER`: Kafka host with port, e.g., `localhost:9092`

## Run the released image

For each release, an image is uploaded to GitHub packages. To download:

```bash
docker pull ghcr.io/alercebroker/correction_step:latest
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
