![unittest](https://github.com/alercebroker/s3_step/workflows/unittest/badge.svg?branch=main) ![integration_test](https://github.com/alercebroker/s3_step/workflows/integration_test/badge.svg?branch=main)
# S3 Upload

## Description

- Upload avro files to bucket in AWS.

#### Previous steps: 
- None

#### Next steps:
- None

## Previous conditions
None

## Version
- **1.0.0:** 
	- Use of APF.
	- Use of boto3.

## Libraries used
- APF
- boto3

## Environment variables

### Consumer setup

- `CONSUMER_TOPICS`: Some topics. String separated by commas. e.g: `topic_one` or `topic_two,topic_three`
- `CONSUMER_SERVER`: Kafka host with port. e.g: `localhost:9092`
- `CONSUMER_GROUP_ID`: Name for consumer group. e.g: `correction`

### Elasticsearch setup
- `ES_PREFIX`: Enables the indexing of term prefixes to speed up prefix searches. e.g: `ztf_pipeline`
- `ES_NETWORK_HOST`: Elasticsearch host.
- `ES_NETWORK_PORT`: Elasticsearch port.

### S3 setup
<<<<<<< Updated upstream
- `BUCKET_NAME`: Mapping of bucket name(s) to topic prefix, e.g., `bucket1:topic1,bucket2:topic2`. The example will send the inputs from topics with names starting with `topic1` to `bucket1` and analogously for `topic2` and `bucket2`.
=======
- `BUCKET_NAME`: Name of bucket to store avro files.
>>>>>>> Stashed changes

### Step metadata
- `STEP_VERSION`: Current version of the step. e.g: `1.0.0`
- `STEP_ID`: Unique identifier for the step. e.g: `S3`
- `STEP_NAME`: Name of the step. e.g: `S3`
- `STEP_COMMENTS`: Comments of the specific version.

### Topic management
For subscribe to specific topi set the following variable:
- `CONSUMER_TOPICS`: Some topics. String separated by commas. e.g: `topic_one` or `topic_two,topic_three`. Or a regular expression like `^topic_.*`

Another way is set a topic strategy, where the topic can change the name. For example in ZTF topics the name of topics is `ztf_<yyyymmdd>_programid1`. How to set up it?
- `TOPIC_STRATEGY_FORMAT`: The topic expression, where `%s` is the date in the string (e.g. `ztf_%s_progamid1`).

### Metrics setup
- `METRICS_HOST`: Kafka host for storing metrics.
- `METRICS_TOPIC`: Name of the topic to store metrics.




## Stream

This step require only consumer.

### Input schema
- [Documentation of ZTF Avro schema.](https://zwickytransientfacility.github.io/ztf-avro-alert/schema.html)
- [Avsc files](https://github.com/ZwickyTransientFacility/ztf-avro-alert/tree/master/schema)

### Output schema
- None

## Run step

### Run container of docker
You can use a `docker run` command, you must set all environment variables.
```bash
docker run --name my_s3_step -e BUCKET_NAME=myhost -e [... all env ...] -d s3_step:version
```

### Run docker-compose
Also you can edit the environment variables in [`docker-compose.yml`](https://github.com/alercebroker/s3_step/blob/main/docker-compose.yml) file. After that use `docker-compose up` command. This run only one container.

```bash
docker-compose up -d
```

If you want scale this container, you must set a number of containers to run.

```bash
docker-compose up -d --scale s3_step=32
```

**Note:** Use `docker-compose down` for stop all containers.

### Run the released image
For each release an image is uploaded to ghcr.io that you can use instead of building your own. To do that replace `docker-compose.yml` or the `docker run` command with this image:

```bash
docker pull ghcr.io/alercebroker/s3_step:latest
```
