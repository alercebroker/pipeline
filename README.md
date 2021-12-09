[![codecov](https://codecov.io/gh/alercebroker/ingestion_step/branch/main/graph/badge.svg?token=S1gZ8mnZAP)](https://codecov.io/gh/alercebroker/ingestion_step)

# Ingestion Step

## Description

Insert data from any survey parsed and assigned by the [Sorting Hat](https://github.com/alercebroker/sorting_hat_step)

This step performs:

- calculate object statistics
- lightcurve correction
- previous candidates processing
- insert objects
- insert detections
- insert non detections

#### Previous steps:

- [Sorting Hat](https://github.com/alercebroker/sorting_hat_step)

#### Next steps:

- None

## Database interactions


### Select:

- Query to get the light curve (detections and non detections).
- Query if object exists in database.

### Insert:

- New detection.
- New non-detection(s).
- Objects

## Previous conditions

No special conditions, only connection to kafka and database.

## Version

- **1.0.1**
https://github.com/alercebroker/ingestion_step/releases/tag/1.0.0-rc3


## Libraries used

- [APF](https://github.com/alercebroker/APF)
- [LC Correction](https://github.com/alercebroker/lc_correction)
- [DB Plugins](https://github.com/alercebroker/db-plugins/releases/tag/2.0.2)

## Environment variables

### DB setup

- `DB_HOST`: Database host for connection.
- `DB_USER`: Database user for read/write (requires these permission).
- `DB_PASSWORD`: Password of user.
- `DB_PORT`: Port connection.
- `DATABASE`: Name of database.

### Consumer setup

- `CONSUMER_TOPICS`: Some topics. String separated by commas. e.g: `topic_one` or `topic_two,topic_three`
- `CONSUMER_SERVER`: Kafka host with port. e.g: `localhost:9092`
- `CONSUMER_GROUP_ID`: Name for consumer group. e.g: `ingestion-step`
- `CONSUME_TIMEOUT`: Max seconds to wait for a message. e.g: `60`
- `CONSUME_MESSAGES`: Ammount of messages to consume for each operation. e.g: `500`
- `TOPIC_STRATEGY_FORMAT` (optional): Topic format to format topics that change every day. e.g: `ztf_{}_programid1`
- `CONSUMER_TOPICS` (optional): Topic list to consume. e.g: `ztf_*`

You must at least use one of `TOPIC_STRATEGY_FORMAT` or `CONSUMER_TOPICS`

### Step metadata

- `STEP_VERSION`: Current version of the step. e.g: `1.0.0`
- `STEP_ID`: Unique identifier for the step. e.g: `S3`
- `STEP_NAME`: Name of the step. e.g: `S3`
- `STEP_COMMENTS`: Comments of the specific version.

## Stream

This step require a consumer.

### Input schema

[Generic Alert](https://github.com/alercebroker/sorting_hat_step/blob/main/schema.py)

### Output schema

[Schema](https://github.com/alercebroker/ingestion_step/blob/1.0.1/schema.py)

## Build docker image

For use this step, first you must build the image of docker. After that you can run the step for use it.

```bash
docker build -t ingestion_step:version .
```

## Run step

### Run container of docker

You can use a `docker run` command, you must set all environment variables.

```bash
docker run --name ingestion_step -e DB_HOST=myhost -e [... all env ...] -d ingestion_step:version
```

### Run docker-compose

Also you can edit the environment variables in [`docker-compose.yml`](https://github.com/alercebroker/ingestion_step/blob/1.0.1/docker-compose.yml) file. After that use `docker-compose up` command. This run only one container.

```bash
docker-compose up -d
```

If you want scale this container, you must set a number of containers to run.

```bash
docker-compose up -d --scale ingestion_step=n
```

**Note:** Use `docker-compose down` for stop all containers.

### Run the released image

For each release an image is uploaded to ghcr.io that you can use instead of building your own. To do that replace docker-compose.yml or the docker run command with this image:

```bash
docker pull ghcr.io/alercebroker/ingestion_step:latest
```

## Local Installation

### Requirements

To install the required packages run

```bash
pip install -r requirements.txt
```

After that you can modify the logic of the step in [step.py](https://github.com/alercebroker/ingestion_step/blob/1.0.1/ingestion_step/step.py) and run 

```
python scripts/run_step.py
```
